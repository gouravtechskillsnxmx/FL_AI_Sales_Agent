# ws_server.py (UPDATED)
import os
import sys
import time
import json
import uuid
import base64
import logging
import tempfile
import asyncio
from urllib.parse import urlparse, parse_qs, quote_plus

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException, ConnectionError

import openai
import boto3
from gtts import gTTS
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# --- Configuration from env ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
HOSTNAME = os.environ.get("HOSTNAME")  # e.g., fl-ai-sales-agent3.onrender.com

# Setup logging
logger = logging.getLogger("fl_ai_sales.ws")
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)

# Create FastAPI app
app = FastAPI()

# Initialize boto3 S3 client
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Twilio client
twilio_client = None
if TWILIO_SID and TWILIO_TOKEN:
    try:
        twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
    except Exception:
        logger.exception("Failed to init Twilio client")

# OpenAI key setup
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# --- Helper functions ---

def make_twiml_response(twiml: str) -> Response:
    """Return TwiML with Content-Type application/xml"""
    return Response(content=twiml.strip(), media_type="application/xml")


def recording_callback_url(request: Request) -> str:
    """
    Build a full callback URL for /recording.
    Prefer HOSTNAME env (external). Otherwise use incoming Host header or fallback.
    """
    try:
        if HOSTNAME:
            return f"https://{HOSTNAME}/recording"
        host = request.headers.get("host")
        scheme = request.url.scheme if hasattr(request, "url") else "https"
        if host:
            return f"{scheme}://{host}/recording"
    except Exception:
        pass
    return "/recording"


def build_download_url(recording_url: str | None) -> str | None:
    """
    Normalize Twilio recording URLs for direct download.
    Twilio provides a Recording resource URL, and the actual audio is at <url>.mp3
    """
    if not recording_url:
        return None

    recording_url = recording_url.strip()

    # Already has extension
    if recording_url.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
        return recording_url

    try:
        parsed = urlparse(recording_url)
        host = (parsed.netloc or "").lower()
        if "api.twilio.com" in host and "/Recordings/" in parsed.path:
            return recording_url + ".mp3"
    except Exception:
        # fallback
        return recording_url

    return recording_url


def create_and_upload_tts(text: str, prefix: str = "tts") -> str:
    """
    Create TTS MP3 (gTTS), upload to S3 under <prefix>/ and return the S3 object key.
    Ensures ContentType: audio/mpeg is set on the uploaded object.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()

    # create mp3 using gTTS
    gTTS(text=text, lang="en").save(tmp_name)

    key = f"{prefix}/{os.path.basename(tmp_name)}"
    try:
        s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
        logger.info("Uploaded TTS to s3://%s/%s", S3_BUCKET, key)
    except Exception:
        logger.exception("Failed to upload TTS to s3://%s/%s", S3_BUCKET, key)
        raise
    return key


def make_proxy_url(s3_key: str) -> str:
    """
    Construct a /tts-proxy URL for Twilio to fetch audio via our app.
    """
    safe_key = quote_plus(s3_key)
    if HOSTNAME:
        return f"https://{HOSTNAME}/tts-proxy?key={safe_key}"
    return f"/tts-proxy?key={safe_key}"

# --- End helpers ---


# --- Endpoints ---
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    logger.info("/twiml hit")
    resp = VoiceResponse()
    resp.say("Hello, this is our AI sales assistant. Please say something after the beep.", voice="alice")
    action_url = recording_callback_url(request)
    resp.record(max_length=5, action=action_url, play_beep=True, timeout=2)
    return make_twiml_response(str(resp))


@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    try:
        form = await request.form()
        payload = dict(form)
    except Exception as e:
        logger.exception("Failed to parse form in /recording: %s", e)
        return make_twiml_response("<Response><Say>Invalid request received.</Say></Response>")

    recording_url = payload.get("RecordingUrl") or payload.get("recordingurl")
    recording_sid = payload.get("RecordingSid") or payload.get("recordingsid")
    call_sid = payload.get("CallSid") or payload.get("callsid")
    from_number = payload.get("From")
    to_number = payload.get("To")

    logger.info("Twilio /recording webhook payload: %s", payload)

    if not call_sid:
        logger.warning("/recording missing CallSid; payload: %s", payload)
        return make_twiml_response("<Response><Say>Missing CallSid in webhook.</Say></Response>")

    # schedule background processing and ack quickly
    background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
    # Return 204 No Content (Twilio accepts this)
    return Response(status_code=204)


@app.get("/tts-proxy")
def tts_proxy(key: str):
    # basic validation
    if not key or ".." in key or key.startswith("/"):
        return JSONResponse({"error":"invalid_key"}, status_code=400)
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body_stream = obj["Body"]
        content_type = meta.get("ContentType", "audio/mpeg")
        return StreamingResponse(body_stream, media_type=content_type)
    except Exception as e:
        logger.exception("tts-proxy failed for key=%s: %s", key, e)
        return JSONResponse({"error":"tts_proxy_failed", "detail": str(e)}, status_code=500)


# --- process_recording_background (proxy-based twiml update) ---
async def process_recording_background(
    call_sid: str,
    recording_url: str | None = None,
    recording_sid: str | None = None,
    payload: dict | None = None
):
    try:
        logger.info("[%s] downloading recording: %s", call_sid, recording_url)
        download_url = build_download_url(recording_url)

        # Ensure we have Twilio credentials for auth when fetching Twilio-hosted recordings
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None

        # Download with retries
        resp = None
        max_retries = 3
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("[%s] download attempt %d: %s", call_sid, attempt, download_url)
                r = requests.get(download_url, auth=auth, timeout=20)
                r.raise_for_status()
                resp = r
                break
            except RequestException as e:
                if isinstance(e, ConnectionError) and ("Failed to resolve" in str(e) or "Name or service not known" in str(e)):
                    logger.error("[%s] name resolution error for %s: %s", call_sid, download_url, e)
                    resp = None
                    break
                logger.warning("[%s] download attempt %d failed: %s", call_sid, attempt, e)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    resp = None

        if resp is None:
            logger.error("[%s] Failed to download recording after %d attempts: %s", call_sid, max_retries, download_url)
            # send fallback TwiML to caller
            try:
                fallback = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>Sorry, I couldn't download your response. Please try again later.</Say></Response>"
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback)
            except Exception:
                logger.exception("[%s] failed to send fallback TwiML", call_sid)
            return

        # Save temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name

        # STT (assumes you have transcribe_with_openai implemented elsewhere)
        transcript = ""
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception:
            logger.exception("[%s] transcribe_with_openai failed", call_sid)

        # Process with your agent (assumes process_user_text exists)
        result = {}
        try:
            if asyncio.iscoroutinefunction(process_user_text):
                result = await process_user_text("default", call_sid, transcript)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))
        except Exception:
            logger.exception("[%s] process_user_text failed", call_sid)

        assistant_text = result.get("reply_text") if isinstance(result, dict) else (str(result) if result is not None else "")
        if not assistant_text:
            assistant_text = "Sorry, I'm having trouble answering right now."

        # Upload TTS and build proxy URL
        try:
            s3_key = create_and_upload_tts(assistant_text)  # returns key like "tts/tmpxxx.mp3"
            proxy_url = make_proxy_url(s3_key)
            if HOSTNAME:
                record_action = f"https://{HOSTNAME}/recording"
            else:
                record_action = "/recording"

            # Build TwiML with XML declaration
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{proxy_url}</Play>
    <Record maxLength="10" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
            logger.info("[%s] updating Twilio with TwiML (proxy): %s", call_sid, twiml)

            try:
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=twiml)
                    logger.info("[%s] Twilio update with proxy succeeded", call_sid)
            except Exception:
                logger.exception("[%s] Twilio update with proxy failed; attempting fallback Say", call_sid)
                try:
                    fallback = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>Sorry, I couldn't play the response. Please try again later.</Say></Response>"
                    if twilio_client:
                        twilio_client.calls(call_sid).update(twiml=fallback)
                except Exception:
                    logger.exception("[%s] sending fallback also failed", call_sid)

        except Exception:
            logger.exception("[%s] create_and_upload_tts or proxy build failed", call_sid)
            try:
                fallback = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>Sorry, I couldn't prepare the reply. Please try again later.</Say></Response>"
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback)
            except Exception:
                logger.exception("[%s] failed to send fallback TwiML after TTS error", call_sid)

    except Exception as e:
        logger.exception("[%s] Unexpected error in process_recording_background: %s", call_sid, e)


# Simple health endpoint
@app.get("/")
def index():
    return PlainTextResponse("OK")
