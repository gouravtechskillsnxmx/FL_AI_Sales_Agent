# ws_server.py
"""
FastAPI app that hosts:
- GET /health
- POST /respond  (calls process_user_text in-process)
- POST /twiml    (returns TwiML to start Twilio Media Stream)
- WS  /twilio/stream  (receives Twilio media events, runs STT -> agent -> TTS -> replies)
"""

import os
import sys
import time
import json
import uuid
import base64
import logging
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# continue in flask_app.py
import os
import requests
import openai
import boto3
from twilio.rest import Client
from gtts import gTTS
import tempfile
from flask import jsonify

# env/config
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")  # bucket must be public-read or use presigned URL
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
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

# Import core business logic (must not import ws_server to avoid cycles)
from langchain_agent_outbound import process_user_text

app = FastAPI()

# Simple request-logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
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
        result = process_user_text(payload.script_id, payload.convo_id, payload.user_text)
        return RespondResponse(
            reply_text=result.get("reply_text", ""),
            used_script=result.get("used_script", True),
            next_state=result.get("next_state", 0)
        )
    except Exception as e:
        logger.exception("Error in /respond: %s", e)
        # fail gracefully
        return RespondResponse(reply_text="[error]", used_script=True, next_state=0)

@app.route("/recording", methods=["POST"])
def recording():
    """
    Twilio will POST recording metadata here after each Record completes.
    Fields include RecordingUrl, CallSid, etc.
    """
    recording_url = request.form.get("RecordingUrl")  # e.g. https://api.twilio.com/2010-04-01/...
    call_sid = request.form.get("CallSid")
    from_number = request.form.get("From")
    to_number = request.form.get("To")

    if not recording_url or not call_sid:
        return ("Missing recording", 400)

    # 1) download recording (mp3 or wav)
    r = requests.get(recording_url + ".mp3")  # Twilio may require adding .mp3
    if r.status_code != 200:
        return ("Failed to download recording", 500)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(r.content)
    tmp.flush()
    tmp.close()

    # 2) send to STT (OpenAI Whisper via /v1/audio/transcriptions)
    transcript = transcribe_with_openai(tmp.name)

    # 3) call OpenAI chat (maintain simple context)
    ctx = CONTEXT.setdefault(call_sid, [{"role": "system", "content": "You are a friendly sales assistant."}])
    ctx.append({"role": "user", "content": transcript})

    chat_resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or gpt-4o, gpt-4o-mini-instruct; choose your model
        messages=ctx,
        max_tokens=150
    )
    assistant_text = chat_resp["choices"][0]["message"]["content"].strip()
    ctx.append({"role": "assistant", "content": assistant_text})

    # 4) make TTS (quick prototype with gTTS) and upload to S3
    tts_url = create_and_upload_tts(assistant_text)

    # 5) instruct Twilio to play the audio into the active call by updating call TwiML
    # Build TwiML to Play the TTS URL and then re-record (loop)
    twiml = f"""<Response>
        <Play>{tts_url}</Play>
        <Record maxLength="5" action="{request.url_root}recording" playBeep="true" timeout="2"/>
    </Response>"""

    try:
        twilio_client.calls(call_sid).update(twiml=twiml)
    except Exception as e:
        print("Twilio update failed:", e)

    return jsonify({"transcript": transcript, "assistant": assistant_text})

def transcribe_with_openai(file_path: str) -> str:
    """
    Use OpenAI Whisper API (speech-to-text batch).
    """
    with open(file_path, "rb") as f:
        resp = openai.Audio.transcribe("gpt-4o-transcribe", file=f)  # or "whisper-1" depending on available models
    return resp["text"].strip()

def create_and_upload_tts(text: str) -> str:
    """
    Prototype: gTTS -> temporary mp3 -> upload to S3 -> return public URL.
    Replace gTTS with ElevenLabs for better quality.
    """
    tts = gTTS(text=text, lang="en")
    tmp_tts = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_tts_name = tmp_tts.name
    tmp_tts.close()
    tts.save(tmp_tts_name)

    key = f"tts/{os.path.basename(tmp_tts_name)}"
    s3.upload_file(tmp_tts_name, S3_BUCKET, key, ExtraArgs={"ACL": "public-read", "ContentType": "audio/mpeg"})
    url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return url



# flask_app.py (add or integrate into your app)
from flask import Flask, request, Response, url_for
from twilio.twiml.voice_response import VoiceResponse

app = Flask(__name__)

@app.route("/twiml", methods=["POST", "GET"])
def twiml():
    # This TwiML says a short intro then records up to 5s of user audio,
    # then POSTs the recording to /recording for processing.
    resp = VoiceResponse()

    # 1) Play or Say your pre-trained marketing script (replace with your script)
    resp.say("Hello, this is Acme sales. I'm calling to share a brief offer.", voice="alice")

    # 2) Record user reply (short)
    # action -> Twilio will POST recording details (RecordingUrl) to /recording
    resp.record(max_length=5, action=url_for('recording', _external=True), play_beep=True, timeout=2)

    # 3) Optionally hang up or loop back to record again after playing reply (we'll control later)
    return Response(str(resp), mimetype="text/xml")

# -------------------------
# Placeholder STT and TTS functions (replace with real providers)
# -------------------------
def stt_from_wav_bytes(wav_bytes: bytes) -> str:
    """
    Replace with real STT (Whisper/Google). For now returns placeholder text.
    """
    # TODO: implement real STT call
    return "[transcript placeholder]"

def tts_synthesize_to_pcm16(text: str) -> bytes:
    """
    Replace with real TTS (ElevenLabs/Polly/Google).
    Must return raw PCM16LE bytes at 16000 Hz mono.
    """
    # 160 ms silence at 16k mono s16le = 0.16 * 16000 samples * 2 bytes = 5120 bytes
    return b"\x00" * 5120

# -------------------------
# ffmpeg-based converters (µ-law <-> PCM16)
# -------------------------
async def mulaw_b64_to_wav_bytes(b64_payload: str) -> bytes:
    mu_bytes = base64.b64decode(b64_payload)
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await proc.communicate(mu_bytes)
    return out

async def pcm16_bytes_to_mulaw_b64(pcm_bytes: bytes) -> str:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
        "-f", "mulaw", "-ar", "8000", "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await proc.communicate(pcm_bytes)
    return base64.b64encode(out).decode("ascii")

# -------------------------
# Twilio Media Streams WebSocket handler
# -------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    # Validate token from query string
    qs = parse_qs(ws.scope.get("query_string", b"").decode())
    token = qs.get("token", [None])[0]
    expected = os.getenv("TWILIO_STREAM_TOKEN", "axcydp34j7")
    if token != expected:
        logger.warning("Invalid TWILIO stream token: %s", token)
        await ws.close(code=4401)
        return

    await ws.accept()
    logger.info("Twilio Media Stream connected")
    stream_sid = None

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"].get("streamSid")
                logger.info("Stream started: %s", stream_sid)

            elif event == "media":
                payload_b64 = msg["media"]["payload"]
                # Convert µ-law -> wav (PCM16k)
                wav_bytes = await mulaw_b64_to_wav_bytes(payload_b64)

                # Run STT (placeholder)
                transcript = stt_from_wav_bytes(wav_bytes)
                logger.info("Transcript (convo=%s): %s", stream_sid or "unknown", transcript)

                # Call in-process agent (fast, no HTTP)
                try:
                    result = process_user_text(script_id="default", convo_id=stream_sid or "unknown", user_text=transcript)
                except Exception as e:
                    logger.exception("process_user_text failed: %s", e)
                    result = {"reply_text": "Sorry, I didn't get that.", "used_script": True, "next_state": 0}

                reply_text = result.get("reply_text", "Sorry, can you repeat?")
                logger.info("Agent reply: %s", reply_text)

                # Synthesize TTS to PCM16 (placeholder)
                pcm_bytes = tts_synthesize_to_pcm16(reply_text)

                # Convert PCM16 -> µ-law base64
                mulaw_b64 = await pcm16_bytes_to_mulaw_b64(pcm_bytes)

                # Send back to Twilio
                out_msg = {"event": "media", "media": {"payload": mulaw_b64}, "streamSid": stream_sid}
                await ws.send_text(json.dumps(out_msg))

                # Send a mark to indicate end-of-playback
                await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": str(uuid.uuid4())}}))

            elif event == "stop":
                logger.info("Stream stopped")
                await ws.close()
                break

            else:
                logger.debug("Unhandled event: %s", event)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception("Error in Twilio WS loop: %s", e)
        try:
            await ws.close()
        except:
            pass
