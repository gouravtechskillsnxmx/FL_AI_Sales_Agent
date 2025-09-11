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

from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

import requests
import openai
import boto3
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from gtts import gTTS

# ---------------- env/config ----------------
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")  # bucket must be public-read or use presigned URL
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
HOSTNAME = os.environ.get("HOSTNAME", "")  # optional: fl-ai-sales-agent3.onrender.com

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
    Fields include RecordingUrl, CallSid, etc.
    We quickly acknowledge (204) and run the heavy work in background.
    """
    form = await request.form()
    recording_url = form.get("RecordingUrl")
    call_sid = form.get("CallSid")
    from_number = form.get("From")
    to_number = form.get("To")

    if not recording_url or not call_sid:
        return JSONResponse({"error": "missing RecordingUrl or CallSid"}, status_code=400)

    # schedule background processing so Twilio isn't waiting
    background_tasks.add_task(process_recording_background, call_sid, recording_url)
    return JSONResponse({}, status_code=204)

# -------------------------
# Background processing pipeline
# -------------------------
async def process_recording_background(call_sid: str, recording_url: str):
    """
    Background pipeline:
    1) download recording (add .mp3)
    2) transcribe (STT) -> transcript text
    3) call process_user_text (agent) -> get reply_text
    4) create TTS -> upload -> get public URL
    5) tell Twilio to Play the audio in the call, then re-record (loop)
    """
    try:
        logger.info("[%s] downloading recording: %s", call_sid, recording_url)
        r = requests.get(recording_url + ".mp3", timeout=30)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.flush()
        tmp.close()
        file_path = tmp.name

        # 2) STT: transcribe file. Replace this with your preferred STT provider.
        transcript = transcribe_with_openai(file_path)
        logger.info("[%s] transcript: %s", call_sid, transcript)

        # 3) call agent
        # process_user_text(script_id, convo_id, user_text)
        if asyncio.iscoroutinefunction(process_user_text):
            result = await process_user_text("default", call_sid, transcript)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))

        assistant_text = result.get("reply_text") if isinstance(result, dict) else str(result)
        if not assistant_text:
            assistant_text = "Sorry, I couldn't understand. Could you repeat that?"

        # 4) TTS: create and upload
        tts_url = create_and_upload_tts(assistant_text)
        logger.info("[%s] tts_url: %s", call_sid, tts_url)

        # 5) Tell Twilio to play audio and then re-enter Record to capture next user utterance
        # Build TwiML: play the TTS and then record again
        record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else f"https://{twilio_client.rest.core.base_uri.split('//')[-1]}/recording"
        twiml = f"""<Response>
            <Play>{tts_url}</Play>
            <Record maxLength="5" action="{record_action}" playBeep="true" timeout="2"/>
        </Response>"""
        try:
            if twilio_client:
                twilio_client.calls(call_sid).update(twiml=twiml)
            else:
                logger.warning("Twilio client not configured; cannot update call %s", call_sid)
        except Exception as e:
            logger.exception("Error updating Twilio call %s: %s", call_sid, e)

    except Exception as e:
        logger.exception("Error in processing recording for %s: %s", call_sid, e)

# -------------------------
# STT and TTS helper functions
# -------------------------
def transcribe_with_openai(file_path: str) -> str:
    """Blocking; uses OpenAI audio transcription endpoint."""
    with open(file_path, "rb") as f:
        # Replace model name as per your available models
        resp = openai.Audio.transcribe("whisper-1", f)
    return resp.get("text", "").strip()

def create_and_upload_tts(text: str) -> str:
    """Synchronous gTTS -> mp3 -> upload to S3 -> return public URL"""
    tts = gTTS(text=text, lang="en")
    tmp_tts = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_tts_name = tmp_tts.name
    tmp_tts.close()
    tts.save(tmp_tts_name)

    key = f"tts/{os.path.basename(tmp_tts_name)}"
    s3.upload_file(tmp_tts_name, S3_BUCKET, key, ExtraArgs={"ACL": "public-read", "ContentType": "audio/mpeg"})
    url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return url

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
