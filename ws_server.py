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

# -------------------------
# TwiML endpoint (returns TwiML with secure wss URL)
# -------------------------
@app.post("/twiml", response_class=Response)
async def twiml():
    token = os.getenv("TWILIO_STREAM_TOKEN", "axcydp34j7")
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "fl-ai-sales-agent3.onrender.com")
    ws_url = f"wss://{host}/twilio/stream?token={token}"
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{ws_url}"/>
  </Start>
  <Say voice="alice">Hello — please hold while I connect you.</Say>
  <Pause length="1"/>
</Response>"""
    return Response(content=twiml_xml, media_type="application/xml")

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
