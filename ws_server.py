# ws_server.py
import asyncio
import base64
import json
import logging
import os
import uuid
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response

# Import your core logic directly (no HTTP call)
# ensure langchain_agent_outbound.py is in the same package/repo and defines process_user_text(...)
from langchain_agent_outbound import process_user_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

app = FastAPI()

# place this after `app = FastAPI()` and before any uses of RespondRequest/RespondResponse

from pydantic import BaseModel
from typing import Optional

class RespondRequest(BaseModel):
    script_id: Optional[str] = "default"
    convo_id: Optional[str] = "global"
    user_text: str

class RespondResponse(BaseModel):
    reply_text: str
    used_script: bool
    next_state: int

@app.post("/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest):
    """
    POST /respond
    Accepts JSON: { "script_id": "...", "convo_id": "...", "user_text": "..." }
    Returns JSON with reply_text, used_script, next_state.
    """
    # Call the in-process agent function (ensure it's imported earlier)
    result = process_user_text(payload.script_id, payload.convo_id, payload.user_text)
    return RespondResponse(
        reply_text=result.get("reply_text", ""),
        used_script=result.get("used_script", True),
        next_state=result.get("next_state", 0)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Simple TwiML endpoint: Twilio fetches this URL to start streaming
# -------------------------------------------------------------------
@app.post("/twiml", response_class=Response)
async def twiml():
    """
    Twilio will request this endpoint when placing a call.
    It returns TwiML that tells Twilio to start a Media Stream to our websocket.
    """
    # Use the env var TWILIO_STREAM_TOKEN if set; otherwise default to axcydp34j7
    token = os.getenv("TWILIO_STREAM_TOKEN", "axcydp34j7")
    # Build a secure wss URL that includes the token as a query param
    ws_url = f"wss://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'fl-ai-sales-agent2.onrender.com')}/twilio/stream?token={token}"
    # You can optionally change the initial <Say> or remove the Pause
    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{ws_url}"/>
  </Start>
  <Say voice="alice">Hello — please hold while I connect you.</Say>
  <Pause length="1"/>
</Response>"""
    return Response(content=twiml_xml, media_type="application/xml")


# -------------------------------------------------------------------
# Placeholder STT + TTS (replace with real providers later)
# -------------------------------------------------------------------
def stt_from_wav_bytes(wav_bytes: bytes) -> str:
    """Replace with real STT (Whisper/Google). For now returns placeholder."""
    # TODO: implement real STT call
    return "[transcript placeholder]"


def tts_synthesize_to_pcm16(text: str) -> bytes:
    """
    Replace with a real TTS provider (ElevenLabs / AWS Polly / Google).
    Must return raw PCM16LE bytes sampled at 16k, mono.
    For now return ~160ms silence.
    """
    return b"\x00" * 5120  # 160ms silence at 16k mono s16le


# -------------------------------------------------------------------
# ffmpeg converters: µ-law <-> PCM16 (used by Twilio Media Streams)
# -------------------------------------------------------------------
async def mulaw_b64_to_wav_bytes(b64_payload: str) -> bytes:
    mu_bytes = base64.b64decode(b64_payload)
    p = await asyncio.create_subprocess_exec(
        "ffmpeg", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
        "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await p.communicate(mu_bytes)
    return out


async def pcm16_bytes_to_mulaw_b64(pcm_bytes: bytes) -> str:
    p = await asyncio.create_subprocess_exec(
        "ffmpeg", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
        "-f", "mulaw", "-ar", "8000", "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await p.communicate(pcm_bytes)
    return base64.b64encode(out).decode("ascii")


# -------------------------------------------------------------------
# WebSocket handler for Twilio Media Streams
# -------------------------------------------------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    # Token validation (very simple): expects ?token=... in the ws URL
    qs = parse_qs(ws.scope.get("query_string", b"").decode())
    token = qs.get("token", [None])[0]
    expected = os.getenv("TWILIO_STREAM_TOKEN", "axcydp34j7")
    if token != expected:
        logger.warning("Invalid token in WS connect: %s", token)
        # close with custom code (unauthorized)
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
                # Incoming audio payload (base64 µ-law 8k)
                payload_b64 = msg["media"]["payload"]

                # Convert µ-law base64 -> wav PCM16k for STT
                wav_bytes = await mulaw_b64_to_wav_bytes(payload_b64)

                # Run STT (replace placeholder with real STT)
                transcript = stt_from_wav_bytes(wav_bytes)
                logger.info("Transcript: %s", transcript)

                # Call the in-process LangChain agent logic (fast, no HTTP)
                try:
                    result = process_user_text(
                        script_id="default",
                        convo_id=stream_sid or "unknown",
                        user_text=transcript
                    )
                except Exception as e:
                    logger.exception("process_user_text failed: %s", e)
                    result = {"reply_text": "Sorry, I didn't understand that.", "used_script": True, "next_state": 0}

                reply_text = result.get("reply_text", "Sorry, can you repeat?")
                logger.info("Agent reply: %s", reply_text)

                # Synthesize TTS to PCM16 (replace with real TTS)
                pcm_bytes = tts_synthesize_to_pcm16(reply_text)

                # Convert PCM16 -> µ-law base64 for Twilio playback
                mulaw_b64 = await pcm16_bytes_to_mulaw_b64(pcm_bytes)

                # Send back to Twilio
                out_msg = {
                    "event": "media",
                    "media": {"payload": mulaw_b64},
                    "streamSid": stream_sid,
                }
                await ws.send_text(json.dumps(out_msg))

                # Send mark to indicate end-of-playback (optional)
                await ws.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": str(uuid.uuid4())}
                }))

            elif event == "stop":
                logger.info("Stream stopped")
                await ws.close()
                break

            else:
                logger.debug("Unhandled event: %s", event)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.exception("Error in WebSocket loop: %s", e)
        try:
            await ws.close()
        except:
            pass
