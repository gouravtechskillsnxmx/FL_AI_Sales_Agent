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
    def transcribe_with_openai(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                resp = openai.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",  # or "whisper-1"
                    file=f
                )
            text = resp.text.strip() if hasattr(resp, "text") else resp.get("text", "").strip()
            if not text:
                logger.warning("transcribe_with_openai: got empty transcript, file=%s", file_path)
            return text
        except Exception as e:
            logger.exception("transcribe_with_openai failed: %s", e)
            return ""


# 2) Safe process_user_text stub (define only if missing)
if "process_user_text" not in globals():

    def process_user_text(model_name: str, call_sid: str, user_text: str) -> dict:
        """
        Use OpenAI to classify intent and generate a short reply.
        Returns a dict with keys: intent, confidence, action, reply_text.

        This function prefers the new openai v1+ client (OpenAI.chat.completions.create).
        If unavailable, it falls back to openai.ChatCompletion (legacy).
        """
        fallback = {
            "intent": "unclear",
            "confidence": 0.5,
            "action": "ask_followup",
            "reply_text": "Sorry, could you repeat? Are you looking to buy a specific product or get recommendations?"
        }

        try:
            if not user_text or not user_text.strip():
                return fallback

            # model selection: allow override via env
            LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
            # safety: keep deterministic outputs
            temperature = 0.0
            max_tokens = 200

            # Construct the strict prompt (model should return JSON only)
            system = (
                "You are a concise outbound sales assistant. Always return a single valid JSON object "
                "and nothing else. Keep reply_text <= 25 words."
            )
            user_prompt = f"""
    Transcript: "{user_text}"

    Context:
    - The product is an ebook/subscription sales flow.
    - Possible intents: ["purchase","info","not_interested","unclear","escalate"].
    - Possible actions: ["ask_followup","answer_with_cta","offer_sms","transfer_human","hangup"].

    Return JSON with keys:
    {{ "intent": "<one of the intents>", "confidence": 0.0-1.0, "action": "<one of the actions>", "reply_text": "Short reply to speak next (<=25 words)" }}
    """

            assistant_text = None

            # Try new client (openai>=1.0.0)
            try:
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=OPENAI_KEY) if globals().get("OPENAI_KEY") else OpenAIClient()
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # resp may be object-like or dict-like; extract content robustly
                try:
                    assistant_text = resp.choices[0].message.content
                except Exception:
                    # dict-like fallback
                    assistant_text = resp["choices"][0]["message"]["content"]
            except Exception as new_exc:
                # If new client not available or it failed, try legacy API (older openai package)
                try:
                    logger.debug("process_user_text: new OpenAI client unavailable/failing, falling back to legacy. err=%s", new_exc)
                    resp = openai.ChatCompletion.create(
                        model=LLM_MODEL,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    assistant_text = resp["choices"][0]["message"]["content"]
                except Exception as legacy_exc:
                    logger.exception("process_user_text: both new and legacy OpenAI calls failed: %s / %s", new_exc, legacy_exc)
                    return fallback

            # At this point assistant_text should be a JSON string (model instructed to return JSON only)
            if not assistant_text:
                logger.warning("[%s] process_user_text: empty assistant_text from model", call_sid)
                return fallback

            # Try to parse JSON strictly
            try:
                parsed = json.loads(assistant_text)
                intent = parsed.get("intent", "unclear")
                confidence = float(parsed.get("confidence", 0.5))
                action = parsed.get("action", "ask_followup")
                reply_text = str(parsed.get("reply_text", "")).strip()
                if not reply_text:
                    return fallback
                return {
                    "intent": intent,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "action": action,
                    "reply_text": reply_text
                }
            except Exception as parse_exc:
                # If the model returned non-JSON, salvage a short reply from the text
                logger.exception("process_user_text: failed to parse JSON from model output: %s; output=%r", parse_exc, assistant_text)
                first_line = assistant_text.strip().splitlines()[0][:250]
                return {
                    "intent": "unclear",
                    "confidence": 0.5,
                    "action": "ask_followup",
                    "reply_text": first_line or fallback["reply_text"]
                }

        except Exception as e:
            logger.exception("process_user_text OpenAI call failed: %s", e)
            return fallback


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
    logger.info("Action_URL %s",action_url)
    resp.record(max_length=5, action=action_url, play_beep=True, timeout=2)
    return make_twiml_response(str(resp))

@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    """
    Robust recording callback:
      - attempt synchronous processing with a short timeout so we can return <Play> immediately.
      - if sync path times out, schedule background processing and return a keep-alive Pause.
      - ensure temp files are not deleted while executor threads still need them.
    """
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

    # Configs (env override)
    try:
        SYNC_TIMEOUT_SECONDS = float(os.environ.get("RECORDING_SYNC_TIMEOUT", "18"))
    except Exception:
        SYNC_TIMEOUT_SECONDS = 18.0
    try:
        KEEPALIVE_SECONDS = int(os.environ.get("KEEPALIVE_SECONDS", "45"))
        if KEEPALIVE_SECONDS < 5:
            KEEPALIVE_SECONDS = 5
        elif KEEPALIVE_SECONDS > 120:
            KEEPALIVE_SECONDS = 120
    except Exception:
        KEEPALIVE_SECONDS = 45

    loop = asyncio.get_running_loop()

    # helper to schedule deletion of a temp file after a future is done
    async def cleanup_when_done(fut, path):
        try:
            await asyncio.wrap_future(fut) if isinstance(fut, asyncio.Future) else None
        except Exception:
            # we don't care about the result, just wait for completion
            pass
        await asyncio.sleep(0.05)  # tiny delay to ensure file handles closed
        try:
            if path and os.path.exists(path):
                os.remove(path)
                logger.debug("cleanup_when_done: removed %s", path)
        except Exception:
            logger.exception("cleanup_when_done failed for %s", path)

    # Synchronous path that runs blocking operations in thread executor.
    async def do_sync_processing():
        file_path = None
        transcribe_future = None
        try:
            if not recording_url:
                logger.warning("[%s] no recording_url for sync path", call_sid)
                return None

            download_url = recording_url if recording_url.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")) else recording_url + ".mp3"
            logger.info("[%s] synchronous processing: downloading %s", call_sid, download_url)

            # download in executor (blocking requests)
            def _download():
                auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None
                r = requests.get(download_url, auth=auth, timeout=15)
                r.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tmp.write(r.content)
                tmp.flush()
                tmp.close()
                return tmp.name

            file_path = await loop.run_in_executor(None, _download)
            logger.info("[%s] synchronous processing: downloaded to %s", call_sid, file_path)

            # transcribe in executor and track its future so we don't delete the file while it runs.
            def _transcribe():
                return transcribe_with_openai(file_path)

            transcribe_future = loop.run_in_executor(None, _transcribe)
            # await with timeout left for other steps
            transcript = await asyncio.wait_for(asyncio.wrap_future(asyncio.ensure_future(transcribe_future)), timeout=max(4, SYNC_TIMEOUT_SECONDS - 6))
            logger.info("[%s] synchronous processing: transcript=%r", call_sid, transcript)

            # agent processing (can be blocking)
            if asyncio.iscoroutinefunction(process_user_text):
                result = await process_user_text("default", call_sid, transcript)
            else:
                result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))

            assistant_text = (result.get("reply_text") if isinstance(result, dict) else str(result)) or ""
            if not assistant_text:
                assistant_text = "Sorry, I couldn't understand that."

            # create tts (blocking) in executor
            s3_key = await loop.run_in_executor(None, lambda: create_and_upload_tts(assistant_text))
            logger.info("[%s] synchronous processing: uploaded tts s3://%s/%s", call_sid, S3_BUCKET, s3_key)

            tts_url = make_proxy_url(s3_key)
            full_tts_url = tts_url if tts_url.startswith("http") else (f"https://{HOSTNAME}{tts_url}" if HOSTNAME else tts_url)
            return full_tts_url

        except asyncio.TimeoutError:
            logger.warning("[%s] synchronous processing timed out", call_sid)
            # Do not delete the file here if transcribe_future is still running; schedule cleanup for later.
            if transcribe_future and not transcribe_future.done() and file_path:
                loop.create_task(cleanup_when_done(transcribe_future, file_path))
            return None
        except Exception as e:
            logger.exception("[%s] synchronous processing failed: %s", call_sid, e)
            # if a future still running, schedule cleanup to avoid leaking temp file
            if transcribe_future and not transcribe_future.done() and file_path:
                loop.create_task(cleanup_when_done(transcribe_future, file_path))
            else:
                # safe to remove now
                try:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    logger.exception("[%s] failed removing temp file %s", call_sid, file_path)
            return None
        finally:
            # If transcribe_future is None or completed, it's safe to remove file here.
            try:
                if (not transcribe_future or (hasattr(transcribe_future, "done") and transcribe_future.done())) and file_path:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug("[%s] removed temp file in finally: %s", call_sid, file_path)
            except Exception:
                logger.exception("[%s] cleanup in finally failed for %s", call_sid, file_path)

    # Try sync processing with timeout
    try:
        tts_play_url = await asyncio.wait_for(do_sync_processing(), timeout=SYNC_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("[%s] overall sync processing timed out after %.1fs", call_sid, SYNC_TIMEOUT_SECONDS)
        tts_play_url = None
    except Exception:
        logger.exception("[%s] unexpected error during sync processing", call_sid)
        tts_play_url = None

    if tts_play_url:
        # Return TwiML to play immediate audio, then record again
        logger.info("[%s] returning immediate <Play> for %s", call_sid, tts_play_url)
        record_action = recording_callback_url(request)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{tts_play_url}</Play>
  <Record maxLength="15" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
        # Fallback: schedule background processing and return keep-alive TwiML
        loop = asyncio.get_running_loop()
        loop.create_task(process_recording_background(call_sid, recording_url, recording_sid, payload))
        logger.info("[%s] scheduled process_recording_background via loop.create_task", call_sid)
        return make_twiml_response(twiml)

    twiml_keepalive = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we process your response.</Say>
  <Pause length="{KEEPALIVE_SECONDS}"/>
</Response>"""
    logger.info("[%s] synchronous path failed/slow — scheduled background task and returning keep-alive (pause %ds).",
                call_sid, KEEPALIVE_SECONDS)
    return make_twiml_response(twiml_keepalive)

    

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
# --- process_recording_background (proxy-based twiml update) ---
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

            # --- Agent (LLM) ---
            try:
                # process_user_text may be sync or async; run accordingly
                if asyncio.iscoroutinefunction(process_user_text):
                    logger.info("[%s] transcript from STT (len=%d): %r", call_sid, len(transcript or ""), transcript)
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

        # --- TTS + upload (create S3 key) ---
        try:
            s3_key = create_and_upload_tts(assistant_text)
            logger.info("[%s] created tts s3 key: %s", call_sid, s3_key)
        except Exception as e:
            logger.exception("[%s] TTS generation failed: %s", call_sid, e)
            s3_key = None

        # --- Build TwiML ---
        record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"

        if s3_key:
            # Convert S3 key -> proxy path and ensure absolute URL for Twilio
            proxy_path = make_proxy_url(s3_key)  # may return absolute if HOSTNAME set
            if proxy_path.startswith("http"):
                play_url = proxy_path
            else:
                # make absolute using HOSTNAME (must be set in env for public access)
                if HOSTNAME:
                    play_url = f"https://{HOSTNAME}{proxy_path if proxy_path.startswith('/') else '/' + proxy_path}"
                else:
                    # best-effort fallback: play whatever proxy_path is (may fail if relative)
                    play_url = proxy_path

            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{play_url}</Play>
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
