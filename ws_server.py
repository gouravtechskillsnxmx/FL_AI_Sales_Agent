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



AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac')

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
    resp.say("Hello, this is our AI sales assistant. Please say something after the beep.", voice="alice")

    action_url = recording_callback_url(request)
    resp.record(max_length=5, action=action_url, play_beep=True, timeout=2)
    return Response(content=str(resp), media_type="text/xml")

# -------------------------
# Recording webhook (Twilio posts after a Record completes)
# -------------------------
@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio will POST recording metadata here after each Record completes.
    We log the full payload, validate required fields, ack quickly (204),
    and schedule background processing for valid inputs.
    """
    # read form data first and make sure variables are always defined
    try:
        form = await request.form()
        payload = dict(form)
    except Exception as e:
        logger.exception("Failed to parse form in /recording: %s", e)
        return JSONResponse({"error": "invalid_form"}, status_code=400)

    # pull fields safely
    recording_url = payload.get("RecordingUrl") or payload.get("recordingurl")
    recording_sid = payload.get("RecordingSid") or payload.get("recordingsid")
    call_sid = payload.get("CallSid") or payload.get("callsid")
    from_number = payload.get("From")
    to_number = payload.get("To")

    # log the raw payload for debugging (but avoid logging secrets)
    logger.info("Twilio /recording webhook payload: %s", payload)

    # validate minimum required fields
    if not call_sid:
        logger.warning("/recording missing CallSid; payload: %s", payload)
        # Return a 400 during debugging; change to 204 to avoid Twilio retry in production if desired.
        return JSONResponse({"error": "missing CallSid"}, status_code=400)

    # schedule background processing and ack immediately
    background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
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
    try:
        logger.info("[%s] downloading recording: %s", call_sid, recording_url)
        download_url = build_download_url(recording_url)

        # Ensure we have Twilio credentials for auth when fetching Twilio-hosted recordings
        if not (TWILIO_SID and TWILIO_TOKEN):
            logger.warning("[%s] TWILIO_SID/TWILIO_TOKEN not set; will try unauthenticated download.", call_sid)

        # Try a few retries for transient DNS issues
        max_retries = 3
        backoff = 1.0
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None
                resp = requests.get(download_url, auth=auth, timeout=15)
                resp.raise_for_status()
                break
            except RequestException as e:
                # If this is a DNS / name resolution issue, log specifically and don't retry many times
                if isinstance(e, ConnectionError) and getattr(e, "args", None):
                    err_str = str(e)
                    if "Name or service not known" in err_str or "Failed to resolve" in err_str:
                        logger.error("[%s] Name resolution failed for host in URL '%s': %s", call_sid, download_url, err_str)
                        # No point retrying if DNS can't resolve - break and abort
                        resp = None
                        break
                # For other transient errors we retry a couple times
                logger.warning("[%s] download attempt %d failed for %s: %s", call_sid, attempt, download_url, e)
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    resp = None

        if resp is None:
            logger.error("[%s] Failed to download recording after %d attempts: %s", call_sid, max_retries, download_url)
            return

        # Save temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(resp.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name

        # STT
        transcript = transcribe_with_openai(file_path)
        logger.info("[%s] transcript: %s", call_sid, transcript)

        # call the agent (sync or async)
        if asyncio.iscoroutinefunction(process_user_text):
            result = await process_user_text("default", call_sid, transcript)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))

        assistant_text = result.get("reply_text") if isinstance(result, dict) else str(result)
        if not assistant_text:
            assistant_text = "Sorry, I couldn't understand. Could you repeat that?"

        # TTS -> upload (presigned URL approach)
        tts_url = create_and_upload_tts(assistant_text)
        logger.info("[%s] tts_url: %s", call_sid, tts_url)

        # Build TwiML and update call
        if HOSTNAME:
            record_action = f"https://{HOSTNAME}/recording"
        else:
            record_action = f"/recording"
        twiml = f"""<Response>
            <Play>{tts_url}</Play>
            <Record maxLength="5" action="{record_action}" playBeep="true" timeout="2"/>
        </Response>"""
        try:
            if twilio_client:
                twilio_client.calls(call_sid).update(twiml=twiml)
        except Exception as e:
            logger.exception("[%s] Error updating Twilio call: %s", call_sid, e)

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


def create_and_upload_tts(text: str, expires_in: int = 600) -> str:
    """
    Create TTS, upload without ACL, return a presigned GET URL valid for `expires_in` seconds.
    Default expiry increased to 600s (10 minutes) to avoid Twilio timing issues.
    """
    from gtts import gTTS
    import tempfile, os, logging

    # 1) Create temp MP3
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()
    tts.save(tmp_name)

    # 2) Upload WITHOUT ACL (buckets with ACLs disabled)
    key = f"tts/{os.path.basename(tmp_name)}"
    try:
        s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
    except Exception:
        logger.exception("Failed to upload %s to %s/%s", tmp_name, S3_BUCKET, key)
        raise

    # 3) Generate presigned GET URL with longer expiry
    try:
        presigned = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': S3_BUCKET, 'Key': key},
            ExpiresIn=expires_in  # seconds
        )
        logger.info("Generated presigned TTS URL (expires in %s s): %s", expires_in, presigned)
        return presigned
    except Exception:
        logger.exception("Failed to create presigned URL for %s/%s", S3_BUCKET, key)
        # fallback (may be private so Twilio could fail)
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"

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
