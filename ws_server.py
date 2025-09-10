import asyncio
import base64
import json
import logging
import os
import uuid
from urllib.parse import parse_qs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Import your core logic directly (no HTTP call)
from langchain_agent_outbound import process_user_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

app = FastAPI()

# -------------------------------------------------------------------
# Placeholder STT + TTS (replace with real APIs later)
# -------------------------------------------------------------------
def stt_from_wav_bytes(wav_bytes: bytes) -> str:
    """Replace this with a real STT provider (e.g., Whisper, Google STT)."""
    # For now, return dummy transcript
    return "[transcript placeholder]"

def tts_synthesize_to_pcm16(text: str) -> bytes:
    """Replace this with a real TTS provider (e.g., ElevenLabs, Polly).
    Must return raw PCM16LE 16k mono bytes.
    """
    # For now, return 160 ms silence at 16kHz (5120 bytes)
    return b"\x00" * 5120

# -------------------------------------------------------------------
# ffmpeg converters: µ-law <-> PCM16
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
    # --- Token validation ---
    qs = parse_qs(ws.scope.get("query_string", b"").decode())
    token = qs.get("token", [None])[0]
    expected = os.getenv("TWILIO_STREAM_TOKEN", "axcydp34j7")
    if token != expected:
        logger.warning("Invalid token in WS connect: %s", token)
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

                # µ-law b64 -> wav (PCM16k)
                wav_bytes = await mulaw_b64_to_wav_bytes(payload_b64)

                # STT (replace with real STT)
                transcript = stt_from_wav_bytes(wav_bytes)
                logger.info("Transcript: %s", transcript)

                # Call core agent logic directly
                result = process_user_text(
                    script_id="default",
                    convo_id=stream_sid or "unknown",
                    user_text=transcript
                )
                reply_text = result.get("reply_text", "Sorry, can you repeat?")
                logger.info("Agent reply: %s", reply_text)

                # TTS (replace with real)
                pcm_bytes = tts_synthesize_to_pcm16(reply_text)

                # PCM16 -> µ-law b64
                mulaw_b64 = await pcm16_bytes_to_mulaw_b64(pcm_bytes)

                # Send back to Twilio
                out_msg = {
                    "event": "media",
                    "media": {"payload": mulaw_b64},
                    "streamSid": stream_sid,
                }
                await ws.send_text(json.dumps(out_msg))

                # Mark end-of-playback
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
