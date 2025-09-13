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

# --- Safe testing stubs and call-status guarded update for process_recording_background ---

# 1) Safe transcribe stub (define only if missing)
if "transcribe_with_openai" not in globals():
    def transcribe_with_openai(audio_path: str) -> str:
        """
        Temporary safe stub for transcription.
        Replace with your production transcription using OpenAI whisper or other STT.
        For now returns empty string (meaning 'no transcript') and logs the event.
        """
        try:
            logger.info("transcribe_with_openai: stub used for file=%s", audio_path)
            # Optionally: if you want basic local transcription for quick tests,
            # integrate a lightweight STT here, otherwise return empty.
            return ""  # empty transcript -> agent fallback will be used
        except Exception as e:
            logger.exception("transcribe_with_openai (stub) failed: %s", e)
            return ""

# 2) Safe process_user_text stub (define only if missing)
if "process_user_text" not in globals():
    def process_user_text(model_name: str, call_sid: str, user_text: str) -> dict:
        """
        Temporary stub for agent processing.
        Replace with your langchain/agent call to produce a reply and optional metadata.
        Returns a dict with "reply_text".
        """
        logger.info("process_user_text (stub) called model=%s call_sid=%s user_text=%r", model_name, call_sid, user_text)
        # Basic echo-ish reply for quick testing:
        if not user_text or user_text.strip() == "":
            # simulate inability to hear user
            return {"reply_text": "Sorry, I had trouble hearing you. Could you repeat that?"}
        else:
            # simple canned answer (replace with GPT call)
            return {"reply_text": f"I heard you say: {user_text}. Thanks for that."}

# 3) Helper to check call status safely
from twilio.base.exceptions import TwilioRestException

def call_is_in_progress(call_sid: str) -> bool:
    """
    Return True if Twilio call resource is currently 'in-progress'.
    If we cannot fetch the call, we return False.
    """
    if not twilio_client:
        logger.warning("call_is_in_progress: no twilio_client available")
        return False
    try:
        call = twilio_client.calls(call_sid).fetch()
        status = getattr(call, "status", None)
        logger.info("[%s] remote call status=%s", call_sid, status)
        return status == "in-progress"
    except TwilioRestException as e:
        logger.warning("[%s] cannot fetch call status: %s", call_sid, e)
        return False
    except Exception:
        logger.exception("[%s] unexpected error fetching call status", call_sid)
        return False

# 4) Update process_recording_background (only the update/redirect part) to check call status
# If your file already contains process_recording_background, edit the section where you call
# `twilio_client.calls(call_sid).update(twiml=twiml)` and replace it with the guarded version below.
#
# I'll provide a small helper function to perform a safe update with call-status check:

def safe_update_call_with_twiml(call_sid: str, twiml: str) -> bool:
    """
    Try to update an in-progress call with new TwiML. Returns True if update succeeded.
    If call is not in-progress, skip update and return False.
    """
    if not twilio_client:
        logger.warning("[%s] safe_update_call_with_twiml: no twilio_client configured", call_sid)
        return False

    try:
        # Avoid updating if call is not active/in-progress
        if not call_is_in_progress(call_sid):
            logger.info("[%s] skipping Twilio update because call is not in-progress.", call_sid)
            return False

        twilio_client.calls(call_sid).update(twiml=twiml)
        logger.info("[%s] safe_update_call_with_twiml: update succeeded", call_sid)
        return True
    except TwilioRestException as e:
        # If Twilio says call cannot be redirected, log and return False.
        logger.warning("[%s] TwilioRestException while updating call: %s", call_sid, e)
        return False
    except Exception:
        logger.exception("[%s] Unexpected error when updating Twilio call", call_sid)
        return False

# 5) Clean-up helper for temp files (optional)
import os
def safe_remove_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.exception("safe_remove_file failed for %s", path)

# 6) Example: how to integrate into process_recording_background
# Replace direct calls to `twilio_client.calls(call_sid).update(twiml=twiml)` with:
#
#     updated = safe_update_call_with_twiml(call_sid, twiml)
#     if not updated:
#         logger.info("[%s] not able to redirect call (skipped or failed)", call_sid)
#
# Also ensure you call safe_remove_file(file_path) after transcription/upload to avoid tmp accumulation.
#
# -------------------------------------------------------------------------
# If you want, I can also produce a full patched copy of your process_recording_background
# with these safe-update and cleanup changes included. Paste "patch process_recording_background"
# if you'd like me to update that function for you automatically.


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

# Transcription helper that supports openai>=1.0.0 and falls back to older clients.
import time

def transcribe_with_openai(file_path: str, model: str = "whisper-1", max_retries: int = 2, retry_delay: float = 1.0) -> str:
    """
    Transcribe an audio file using OpenAI.

    Uses the new 'OpenAI' client (openai>=1.0.0) if available:
        from openai import OpenAI
        client = OpenAI()
        resp = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    Falls back to older openai.Audio.transcribe interface if the new one is not present.

    Returns the transcribed text (empty string on failure).
    """
    if not file_path or not os.path.exists(file_path):
        logger.warning("transcribe_with_openai: file not found: %s", file_path)
        return ""

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Try new OpenAI client first (openai>=1.0.0)
            try:
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAIClient()
                with open(file_path, "rb") as f:
                    # `audio.transcriptions.create` returns a response object with `.text` or ['text']
                    resp = client.audio.transcriptions.create(model=model, file=f)
                    # resp may be a dict-like or an object. Try common access patterns:
                    if isinstance(resp, dict):
                        text = resp.get("text") or resp.get("data") or ""
                    else:
                        # object with attribute 'text' in many examples
                        text = getattr(resp, "text", None) or (resp.get("text") if hasattr(resp, "get") else None) or ""
                    if not text:
                        # Some variants return resp['text'] nested. Try stringifying resp.
                        try:
                            text = str(resp)
                        except Exception:
                            text = ""
                    logger.info("transcribe_with_openai: new-client success (len=%d)", len(text) if text else 0)
                    return (text or "").strip()
            except Exception as new_client_exc:
                # If the new client isn't available or raises, try the legacy API
                last_exc = new_client_exc
                logger.debug("transcribe_with_openai: new OpenAI client failed (attempt %d): %s", attempt, new_client_exc)

            # Fallback: older openai package interface (pre-1.0)
            try:
                # legacy `openai` module (old interface) may already be imported as `openai`
                if hasattr(openai, "Audio") and hasattr(openai.Audio, "transcribe"):
                    with open(file_path, "rb") as f:
                        resp = openai.Audio.transcribe(model, f)
                        # resp likely has 'text' attribute or key
                        if isinstance(resp, dict):
                            text = resp.get("text", "") or resp.get("data", "")
                        else:
                            text = getattr(resp, "text", "") or ""
                        logger.info("transcribe_with_openai: legacy-client success (len=%d)", len(text) if text else 0)
                        return (text or "").strip()
                else:
                    # No transcribe interface available
                    raise RuntimeError("No supported OpenAI transcription client available")
            except Exception as legacy_exc:
                last_exc = legacy_exc
                logger.debug("transcribe_with_openai: legacy client failed (attempt %d): %s", attempt, legacy_exc)
                # continue to retry loop below
                raise legacy_exc

        except Exception as exc:
            logger.warning("transcribe_with_openai attempt %d/%d failed: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                logger.exception("transcribe_with_openai: all attempts failed; returning empty transcript.")
                return ""

    # If we somehow exit loop without returning, return empty string
    logger.warning("transcribe_with_openai: reached end without transcription for %s", file_path)
    return ""



# --- process_recording_background (proxy-based twiml update) ---
async def process_recording_background(
    call_sid: str,
    recording_url: str | None = None,
    recording_sid: str | None = None,
    payload: dict | None = None
):
    """
    Background task: download recording, transcribe, call agent, generate TTS, and update Twilio call.
    This version is safe: it always produces TwiML and checks call status before updating.
    """
    file_path = None
    try:
        logger.info("[%s] downloading recording: %s", call_sid, recording_url)
        download_url = f"{recording_url}.mp3" if recording_url else None

        # --- Download recording from Twilio ---
        resp = None
        if download_url:
            auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None
            try:
                resp = requests.get(download_url, auth=auth, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                logger.exception("[%s] Failed to download recording: %s", call_sid, e)

        if not resp or not resp.content:
            logger.warning("[%s] No audio downloaded; using fallback text", call_sid)
            assistant_text = "Sorry, I could not hear anything that time. Please try again."
        else:
            # Save to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(resp.content)
            tmp.flush()
            tmp.close()
            file_path = tmp.name

            # --- Transcription ---
            try:
                transcript = transcribe_with_openai(file_path)
            except Exception as e:
                logger.exception("[%s] transcribe_with_openai failed: %s", call_sid, e)
                transcript = ""

            # --- Agent ---
            try:
                if asyncio.iscoroutinefunction(process_user_text):
                    result = await process_user_text("default", call_sid, transcript)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))
                assistant_text = result.get("reply_text") if isinstance(result, dict) else str(result)
            except Exception as e:
                logger.exception("[%s] process_user_text failed: %s", call_sid, e)
                assistant_text = "I had trouble processing that. Could you repeat?"

        # Ensure non-empty assistant_text
        if not assistant_text:
            assistant_text = "Sorry, I couldn't understand that. Please try again."

        # --- TTS + upload ---
        try:
            tts_url = create_and_upload_tts(assistant_text)
            logger.info("[%s] tts_url: %s", call_sid, tts_url)
        except Exception as e:
            logger.exception("[%s] TTS generation failed: %s", call_sid, e)
            tts_url = None

        # --- Build TwiML ---
        record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"
        if tts_url:
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{tts_url}</Play>
    <Record maxLength="10" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
        else:
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, something went wrong generating audio. Please try again.</Say>
    <Record maxLength="10" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""

        # --- Update Twilio if call still active ---
        updated = safe_update_call_with_twiml(call_sid, twiml)
        if not updated:
            logger.info("[%s] Call not updated (not in-progress or failed).", call_sid)

    except Exception as e:
        logger.exception("[%s] Unexpected error in process_recording_background: %s", call_sid, e)
    finally:
        safe_remove_file(file_path)


# Simple health endpoint
@app.get("/")
def index():
    return PlainTextResponse("OK")
