# audio_helpers.py
import os
import tempfile
import boto3
import openai
from gtts import gTTS

openai.api_key = os.environ.get("OPENAI_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

def transcribe_with_openai_whisper(file_path: str) -> str:
    """Blocking call to OpenAI audio transcription (Whisper-like)."""
    with open(file_path, "rb") as f:
        # Model name may vary by account: "whisper-1" or "gpt-4o-transcribe"
        resp = openai.Audio.transcribe("whisper-1", f)
    return resp.get("text", "").strip()

def create_tts_and_upload_gtts(text: str) -> str:
    """
    Quick prototype using gTTS and uploading to S3. Returns public HTTPS URL.
    Replace with ElevenLabs or Polly for production quality & latency improvements.
    """
    # create mp3
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()
    tts = gTTS(text=text, lang='en')
    tts.save(tmp_name)

    # upload to S3
    key = f"tts/{os.path.basename(tmp_name)}"
    s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ACL": "public-read", "ContentType": "audio/mpeg"})
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return public_url

# If you want async versions for use in aiohttp background tasks:
import asyncio
async def transcribe_file_async(path):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, transcribe_with_openai_whisper, path)

async def create_tts_and_upload_async(text):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, create_tts_and_upload_gtts, text)
