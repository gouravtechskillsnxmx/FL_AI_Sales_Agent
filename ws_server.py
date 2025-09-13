# ws_server.py
"""
FastAPI app that hosts:
- GET /health
- POST /respond  (calls process_user_text in-process)
- POST /twiml    (returns TwiML to start Twilio Record/Media Stream)
- POST /recording (Twilio posts short recordings here -> STT -> agent -> TTS -> play back)
- (Optional) commented WebSocket handler for Twilio Media Streams (kept as docstring for later)
"""

import os
import sys
import time
import json
import uuid
import base64
import logging
import tempfile
import asyncio
from urllib.parse import urlparse, parse_qs

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException, ConnectionError, HTTPError, Timeout

import openai
import boto3
from gtts import gTTS
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Temporary debug endpoint — add to ws_server.py and redeploy, then call /debug/s3check?key=tts/<filename>
from fastapi import Query
from fastapi.responses import StreamingResponse
from fastapi.responses import Response

import time
from urllib.parse import urlparse, quote_plus
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException, ConnectionError



AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac')


def make_twiml_response(twiml: str) -> Response:
    return Response(content=twiml.strip(), media_type="application/xml")

def build_download_url(recording_url: str) -> str:
    """
    Return a safe download URL.
    If Twilio gives a base RecordingUrl (no extension), Twilio usually requires adding .mp3
    however many external test URLs already include an extension; avoid double-appending.
    """
    if not recording_url:
        return recording_url
    parsed = urlparse(recording_url)
    path = parsed.path.lower()
    # If path already ends with an audio extension, leave it
    if any(path.endswith(ext) for ext in AUDIO_EXTENSIONS):
        return recording_url
    # Otherwise, default to appending .mp3 (Twilio recording URLs often require this)
    return recording_url + ".mp3"




# ---------------- env/config ----------------
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")  # bucket must be public-read or use presigned URL
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
HOSTNAME = os.environ.get("HOSTNAME", "")  # optional: fl-ai-sales-agent3.onrender.com


# Replace old openai import / api_key usage with the new OpenAI client
try:
    # new style (openai>=1.0.0)
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()
except Exception:
    # fallback for older installs (rare)
    import openai as _old_openai
    _old_openai.api_key = OPENAI_KEY
    openai_client = None


# init clients
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None
openai.api_key = OPENAI_KEY
s3 = boto3.client("s3", region_name=AWS_REGION)

# basic in-memory context per call (persist to DB later)
CONTEXT = {}

# Configure logging (stdout so Render captures it)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("fl_ai_sales.ws")

# Import core business logic (your LangChain agent)
# This module must expose process_user_text(script_id, convo_id, user_text)
from langchain_agent_outbound import process_user_text

app = FastAPI()

# Simple request-logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, duration_ms)
        return response

app.add_middleware(RequestLoggingMiddleware)

@app.get("/debug/s3check")
def debug_s3_check(key: str = Query(..., description="S3 object key, e.g. tts/tmpxxxx.mp3")):
    """
    Returns head_object metadata, tries to get_object, and returns a presigned URL.
    Only use temporarily for debugging.
    """
    try:
        # assumes `s3` boto3 client is already created with the app's env creds
        meta = s3.head_object(Bucket=S3_BUCKET, Key=key)
        presigned = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=600
        )
        # try a minimal get_object to check permission (do NOT return full bytes)
        got = s3.get_object(Bucket=S3_BUCKET, Key=key)
        length = got.get("ContentLength")
        ct = meta.get("ContentType")
        return {
            "status": "ok",
            "head_object": {"ContentType": ct, "ContentLength": length},
            "presigned_url": presigned
        }
    except Exception as e:
        # return the exception text so we can see AWS error (AccessDenied, SignatureDoesNotMatch, etc)
        import traceback
        tb = traceback.format_exc()
        return {"status": "error", "error": str(e), "trace": tb}


# -------------------------
# Health endpoint
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------
# Respond endpoint (for manual testing: Postman/curl)
# -------------------------
class RespondRequest(BaseModel):
    script_id: str = "default"
    convo_id: str = "global"
    user_text: str

class RespondResponse(BaseModel):
    reply_text: str
    used_script: bool
    next_state: int

@app.post("/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest):
    try:
        # If process_user_text is sync, run in threadpool to avoid blocking event loop
        if asyncio.iscoroutinefunction(process_user_text):
            result = await process_user_text(payload.script_id, payload.convo_id, payload.user_text)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: process_user_text(payload.script_id, payload.convo_id, payload.user_text))

        return RespondResponse(
            reply_text=result.get("reply_text", ""),
            used_script=result.get("used_script", True),
            next_state=result.get("next_state", 0)
        )
    except Exception as e:
        logger.exception("Error in /respond: %s", e)
        return RespondResponse(reply_text="[error]", used_script=True, next_state=0)

# -------------------------
# Helper: build recording callback URL
# -------------------------
def recording_callback_url(request: Request) -> str:
    """
    Build absolute HTTPS recording callback URL.
    Prefer HOSTNAME env var (recommended). If HOSTNAME is not set,
    fallback to request.url but force https scheme.
    """
    host = HOSTNAME.strip() if HOSTNAME else None
    if host:
        return f"https://{host}/recording"
    # fallback: construct from request but force https
    base = str(request.base_url).rstrip("/")
    # request.base_url may contain http when behind proxies, so force https
    if base.startswith("http://"):
        base = "https://" + base.split("://", 1)[1]
    return f"{base}/recording"


# -------------------------
# TwiML endpoint
# -------------------------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    """Return TwiML to start the call: say intro then record short user reply"""
    resp = VoiceResponse()
    resp.say(
        "Hello, this is our AI sales assistant. Please say something after the beep.",
        voice="alice"
    )

    action_url = recording_callback_url(request)
    resp.record(max_length=5, action=action_url, play_beep=True, timeout=2)

    # Twilio wants `application/xml` here
    return Response(content=str(resp), media_type="application/xml")
# -------------------------
# Recording webhook (Twilio posts after a Record completes)
# -------------------------

def make_twiml_response(twiml: str) -> Response:
    """
    Return TwiML with the exact content-type Twilio expects.
    Use this when you want to return TwiML XML immediately.
    """
    return Response(content=twiml.strip(), media_type="application/xml")

@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio POSTs recording metadata here after each <Record> completes.
    We:
      - parse and log the full payload,
      - validate minimal fields,
      - enqueue background processing, and
      - ACK quickly with 204 (or return TwiML when helpful).
    """
    # read form data first and make sure variables are always defined
    try:
        form = await request.form()
        payload = dict(form)
    except Exception as e:
        logger.exception("Failed to parse form in /recording: %s", e)
        # return TwiML apology so Twilio receives application/xml (helps avoid 12300)
        return make_twiml_response("<Response><Say>Invalid request received.</Say></Response>")

    # pull fields safely (support various casings)
    recording_url = payload.get("RecordingUrl") or payload.get("recordingurl")
    recording_sid = payload.get("RecordingSid") or payload.get("recordingsid")
    call_sid = payload.get("CallSid") or payload.get("callsid")
    from_number = payload.get("From")
    to_number = payload.get("To")

    logger.info("Twilio /recording webhook payload: %s", payload)

    # If CallSid missing, return TwiML so Twilio sees a proper XML response (and won't treat it as an app error)
    if not call_sid:
        logger.warning("/recording missing CallSid; payload: %s", payload)
        return make_twiml_response("<Response><Say>Missing CallSid in webhook.</Say></Response>")

    # If RecordingUrl missing, still ack but inform caller (optional)
    if not recording_url:
        logger.warning("[%s] /recording missing RecordingUrl; payload: %s", call_sid, payload)
        # ACK quickly, but respond with TwiML to be explicit
        background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
        return make_twiml_response("<Response><Say>We did not receive any audio. Please try again.</Say></Response>")

    # schedule background processing and ack immediately (204 No Content)
    background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
    # A 204 is fine here — Twilio will treat it as success. It has no body, so no Content-Type is required.
    return Response(status_code=204)


# -------------------------
# Background processing pipeline
# -------------------------

async def process_recording_background(
    call_sid: str,
    recording_url: str | None = None,
    recording_sid: str | None = None,
    payload: dict | None = None
):
    """
    Robust background pipeline:
      1) download Twilio recording (with auth when needed)
      2) transcribe via OpenAI (transcribe_with_openai)
      3) call agent (process_user_text)
      4) create TTS file and upload to S3 (create_and_upload_tts)
      5) send Twilio TwiML pointing at /tts-proxy?key=...
      6) fallback to <Say> on any failure so caller doesn't hear generic app error
    """
    try:
        logger.info("[%s] background job started. recording_url=%s recording_sid=%s", call_sid, recording_url, recording_sid)

        # Build download URL (your helper)
        download_url = build_download_url(recording_url)
        auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN and "api.twilio.com" in (download_url or "")) else None

        # Download with retries
        resp = None
        max_retries = 3
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("[%s] attempting download (attempt %d): %s (auth=%s)", call_sid, attempt, download_url, bool(auth))
                r = requests.get(download_url, auth=auth, timeout=20)
                r.raise_for_status()
                resp = r
                break
            except RequestException as e:
                # DNS-like fatal
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
            logger.error("[%s] failed to download recording; sending apology and exiting", call_sid)
            fallback = "<Response><Say>Sorry, I couldn't record your response. Please try again later.</Say></Response>"
            try:
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback)
            except Exception:
                logger.exception("[%s] failed to send fallback TwiML", call_sid)
            return

        # Save recording to tmp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name
        logger.info("[%s] saved recording to %s", call_sid, file_path)

        # STT: transcribe_with_openai(file_path) -> returns text
        try:
            transcript = transcribe_with_openai(file_path)
            logger.info("[%s] transcript: %s", call_sid, transcript)
        except Exception:
            logger.exception("[%s] transcribe_with_openai failed", call_sid)
            transcript = ""

        # Agent: process_user_text -> returns dict or string
        try:
            if asyncio.iscoroutinefunction(process_user_text):
                result = await process_user_text("default", call_sid, transcript)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))
        except Exception:
            logger.exception("[%s] process_user_text failed", call_sid)
            result = {}

        assistant_text = result.get("reply_text") if isinstance(result, dict) else (str(result) if result is not None else "")
        if not assistant_text:
            assistant_text = "Sorry, I'm having trouble answering right now."

        # TTS: create_and_upload_tts should upload the MP3 to S3 and return the S3 URL (not required presigned).
        try:
            tts_s3_url = create_and_upload_tts(assistant_text)  # must return e.g. https://bucket.s3.amazonaws.com/tts/tmp....mp3
            logger.info("[%s] create_and_upload_tts returned: %s", call_sid, tts_s3_url)
        except Exception:
            logger.exception("[%s] create_and_upload_tts failed", call_sid)
            tts_s3_url = None

        if not tts_s3_url:
            fallback = "<Response><Say>Sorry, I'm unable to prepare a reply right now.</Say></Response>"
            try:
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback)
            except Exception:
                logger.exception("[%s] failed to send fallback TwiML (no tts)", call_sid)
            return

        # Extract S3 key and build proxy URL (URL-encode key)
        parsed = urlparse(tts_s3_url)
        s3_key = parsed.path.lstrip("/")  # e.g. "tts/tmpabc.mp3"
        proxy_key = quote_plus(s3_key)
        if HOSTNAME:
            proxy_url = f"https://{HOSTNAME}/tts-proxy?key={proxy_key}"
            record_action = f"https://{HOSTNAME}/recording"
        else:
            proxy_url = f"/tts-proxy?key={proxy_key}"
            record_action = "/recording"

        # Build TwiML using proxy_url
        twiml = f"""<Response>
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
                fallback = "<Response><Say>Sorry, I couldn't play the response. Please try again later.</Say></Response>"
                if twilio_client:
                    twilio_client.calls(call_sid).update(twiml=fallback)
            except Exception:
                logger.exception("[%s] sending fallback also failed", call_sid)

    except Exception as e:
        logger.exception("[%s] Unexpected error in process_recording_background: %s", call_sid, e)

# -------------------------
# STT and TTS helper functions
# -------------------------
def transcribe_with_openai(file_path: str) -> str:
    """
    Transcribe audio file using OpenAI new Python client (openai>=1.0.0).
    If you prefer the old client, pin openai==0.28 and restore the old call.
    """
    # Prefer the new client if available
    if 'openai_client' in globals() and openai_client:
        # choose model: "gpt-4o-transcribe" or "whisper-1" depending on availability
        model_name = "gpt-4o-transcribe"  # or "whisper-1" if you want whisper
        with open(file_path, "rb") as audio_f:
            resp = openai_client.audio.transcriptions.create(
                model=model_name,
                file=audio_f
            )
        # resp typically has a .text attribute or a 'text' key
        text = getattr(resp, "text", None)
        if text is None:
            try:
                text = resp.get("text")  # dict-like fallback
            except Exception:
                text = str(resp)
        return text.strip() if isinstance(text, str) else str(text)

    # Fallback to old openai package API (if you pinned openai==0.28)
    try:
        import openai as _old_openai
        with open(file_path, "rb") as f:
            # older API: openai.Audio.transcribe(...)
            resp = _old_openai.Audio.transcribe("whisper-1", f)
        return resp.get("text", "").strip()
    except Exception as e:
        # provide a helpful error in logs
        logger.exception("No OpenAI client available or transcription failed: %s", e)
        raise


def create_and_upload_tts(text: str, expires_in: int = 3600) -> str:
    """
    Create TTS, upload to S3 with proper ContentType, and return a presigned URL
    valid for expires_in seconds (default 1 hour).
    """
    import os, tempfile
    from gtts import gTTS

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()
    gTTS(text=text, lang="en").save(tmp_name)

    key = f"tts/{os.path.basename(tmp_name)}"
    # Upload with explicit ContentType
    s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
    logger.info("Uploaded TTS to s3://%s/%s", S3_BUCKET, key)

    # Generate presigned URL with longer expiry
    presigned = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': S3_BUCKET, 'Key': key},
        ExpiresIn=expires_in
    )
    logger.info("Generated presigned TTS URL (expires_in=%s): %s", expires_in, presigned)
    return presigned


@app.get("/tts-proxy")
def tts_proxy(key: str):
    # basic validation
    if not key or ".." in key or key.startswith("/"):
        return JSONResponse({"error":"invalid_key"}, status_code=400)
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body_stream = obj['Body']
        content_type = meta.get("ContentType", "audio/mpeg")
        return StreamingResponse(body_stream, media_type=content_type)
    except Exception as e:
        logger.exception("tts-proxy failed for key=%s: %s", key, e)
        return JSONResponse({"error":"tts_proxy_failed", "detail": str(e)}, status_code=500)

# -------------------------
# (Optional) Twilio Media Streams WebSocket handler (kept as example)
# To use streaming replace the Record flow above and enable this handler.
# -------------------------
"""
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio Media Stream connected")
    stream_sid = None
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            event = msg.get("event")
            if event == "start":
                stream_sid = msg["start"].get("streamSid")
                logger.info("Stream started: %s", stream_sid)
            elif event == "media":
                payload_b64 = msg["media"]["payload"]
                # Convert µ-law -> wav (PCM16k)
                wav_bytes = await mulaw_b64_to_wav_bytes(payload_b64)
                transcript = stt_from_wav_bytes(wav_bytes)
                # call agent
                result = process_user_text(script_id="default", convo_id=stream_sid or "unknown", user_text=transcript)
                reply_text = result.get("reply_text", "Sorry, can you repeat?")
                # synthesize & convert back to mulaw (omitted here)
                # send back via websocket
            elif event == "stop":
                logger.info("Stream stopped")
                await websocket.close()
                break
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception("Error in Twilio WS loop: %s", e)
        try:
            await websocket.close()
        except:
            pass
"""

# -------------------------
# Run if executed directly (useful for local dev)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ws_server:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
