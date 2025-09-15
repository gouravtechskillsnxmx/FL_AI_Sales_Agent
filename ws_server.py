# ws_server.py
# Combined file: Twilio voice + OpenAI + Agentic runner + Amazon PA-API product search tool
import os
import sys
import time
import json
import uuid
import logging
import tempfile
import asyncio
from urllib.parse import quote_plus

import requests
from requests.auth import HTTPBasicAuth

import boto3
import botocore
from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth
from botocore.credentials import Credentials as BotocoreCredentials

import openai
from gtts import gTTS
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from twilio.base.exceptions import TwilioRestException

from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse

# ---------- Configuration ----------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
HOSTNAME = os.environ.get("HOSTNAME")  # public host (e.g. my-app.onrender.com)

# Amazon PA-API env vars (required for real product search)
AMAZON_PA_ACCESS_KEY = os.environ.get("AMAZON_PA_ACCESS_KEY")
AMAZON_PA_SECRET_KEY = os.environ.get("AMAZON_PA_SECRET_KEY")
AMAZON_PA_PARTNER_TAG = os.environ.get("AMAZON_PA_PARTNER_TAG")
AMAZON_PA_REGION = os.environ.get("AMAZON_PA_REGION", "us-east-1")   # e.g., us-east-1
AMAZON_PA_HOST = os.environ.get("AMAZON_PA_HOST", "webservices.amazon.com")  # default host for US

# ---------- Logging ----------
logger = logging.getLogger("fl_ai_sales.ws")
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)

# ---------- FastAPI app ----------
app = FastAPI()

# ---------- AWS S3 client ----------
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ---------- Twilio client ----------
twilio_client = None
if TWILIO_SID and TWILIO_TOKEN:
    try:
        twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
    except Exception:
        logger.exception("Failed to init Twilio client")

# ---------- OpenAI setup ----------
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ---------- Utilities ----------
def make_twiml_response(twiml: str) -> Response:
    return Response(content=twiml.strip(), media_type="application/xml")

def recording_callback_url(request: Request) -> str:
    if HOSTNAME:
        return f"https://{HOSTNAME}/recording"
    host = request.headers.get("host")
    scheme = request.url.scheme if hasattr(request, "url") else "https"
    if host:
        return f"{scheme}://{host}/recording"
    return "/recording"

def safe_remove_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug("Removed tmp file %s", path)
    except Exception:
        logger.exception("safe_remove_file failed for %s", path)

def build_download_url(recording_url: str | None) -> str | None:
    if not recording_url:
        return None
    if recording_url.lower().endswith((".mp3", ".wav", ".ogg", ".m4a")):
        return recording_url
    return recording_url + ".mp3"

def create_and_upload_tts(text: str, prefix: str = "tts") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_name = tmp.name
    tmp.close()
    gTTS(text=text, lang="en").save(tmp_name)
    key = f"{prefix}/{os.path.basename(tmp_name)}"
    try:
        s3.upload_file(tmp_name, S3_BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})
        logger.info("Uploaded TTS to s3://%s/%s", S3_BUCKET, key)
    except Exception:
        logger.exception("Failed to upload TTS to s3://%s/%s", S3_BUCKET, key)
        raise
    finally:
        # remove local tmp file after upload
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            logger.exception("Failed removing local tmp tts file %s", tmp_name)
    return key

def make_proxy_url(s3_key: str) -> str:
    safe_key = quote_plus(s3_key)
    if HOSTNAME:
        return f"https://{HOSTNAME}/tts-proxy?key={safe_key}"
    return f"/tts-proxy?key={safe_key}"

def call_is_in_progress(call_sid: str) -> bool:
    if not twilio_client:
        logger.warning("call_is_in_progress: no twilio_client")
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
        logger.exception("[%s] error fetching call status", call_sid)
        return False

def safe_update_call_with_twiml(call_sid: str, twiml: str) -> bool:
    if not twilio_client:
        logger.warning("[%s] safe_update_call_with_twiml: no twilio client", call_sid)
        return False
    try:
        if not call_is_in_progress(call_sid):
            logger.info("[%s] skipping Twilio update because call is not in-progress.", call_sid)
            return False
        twilio_client.calls(call_sid).update(twiml=twiml)
        logger.info("[%s] safe_update_call_with_twiml: update succeeded", call_sid)
        return True
    except TwilioRestException as e:
        logger.warning("[%s] TwilioRestException updating call: %s", call_sid, e)
        return False
    except Exception:
        logger.exception("[%s] unexpected error updating call", call_sid)
        return False

# ---------- Transcription (OpenAI) ----------
def transcribe_with_openai(file_path: str, model: str = "whisper-1", max_retries: int = 2, retry_delay: float = 1.0) -> str:
    if not file_path or not os.path.exists(file_path):
        logger.warning("transcribe_with_openai: file missing %s", file_path)
        return ""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # prefer new client
            try:
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAIClient()
                with open(file_path, "rb") as f:
                    resp = client.audio.transcriptions.create(model=model, file=f)
                text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None) or ""
                logger.info("transcribe_with_openai: new-client success (len=%d)", len(text or ""))
                return (text or "").strip()
            except Exception as new_exc:
                last_exc = new_exc
                logger.debug("transcribe_with_openai: new client failed: %s", new_exc)
            # fallback to legacy openai if available
            if hasattr(openai, "Audio") and hasattr(openai.Audio, "transcribe"):
                with open(file_path, "rb") as f:
                    resp = openai.Audio.transcribe(model, f)
                    text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None) or ""
                    logger.info("transcribe_with_openai: legacy-client success (len=%d)", len(text or ""))
                    return (text or "").strip()
            raise RuntimeError("No supported transcription client")
        except Exception as exc:
            logger.warning("transcribe_with_openai attempt %d failed: %s", attempt, exc)
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                logger.exception("transcribe_with_openai: all attempts failed")
                return ""
    return ""

# ---------- Amazon PA-API product search tool (production-ready) ----------
# Uses botocore SigV4 signing to call PA-API v5 (SearchItems).
# Env required: AMAZON_PA_ACCESS_KEY, AMAZON_PA_SECRET_KEY, AMAZON_PA_PARTNER_TAG
def _paapi_signed_post(path: str, body: dict, region: str, host: str, access_key: str, secret_key: str, service: str = "ProductAdvertisingAPI"):
    """
    Build and sign a POST request to Amazon PA-API using botocore SigV4 and return (status_code, json_or_text)
    """
    url = f"https://{host}{path}"
    body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {
        "content-type": "application/json; charset=UTF-8",
        "host": host,
        "x-amz-target": "",  # PA-API doesn't require this header; left blank as safe placeholder
    }
    # Build AWSRequest for signing
    aws_req = AWSRequest(method="POST", url=url, data=body_json, headers=headers)
    creds = BotocoreCredentials(access_key, secret_key)
    signer = SigV4Auth(creds, service, region)
    signer.add_auth(aws_req)
    # prepare headers including Authorization and x-amz-date created by signer
    prepared = aws_req.prepare()
    req_headers = dict(prepared.headers)
    try:
        resp = requests.post(url, data=body_json, headers=req_headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text
    except Exception as e:
        logger.exception("PA-API signed POST failed: %s", e)
        raise

def tool_search_products_pa(query: str, limit: int = 3, locale: str = "us"):
    """
    Call Amazon Product Advertising API v5 SearchItems endpoint.
    Returns a list of normalized product dicts or falls back to mocked results if PA-API not configured/failed.
    """
    q = str(query or "").strip()
    logger.info("tool_search_products_pa: query=%r limit=%d", q, limit)
    if not q:
        return []
    access_key = AMAZON_PA_ACCESS_KEY
    secret_key = AMAZON_PA_SECRET_KEY
    partner_tag = os.environ.get("AMAZON_PA_PARTNER_TAG")
    region = AMAZON_PA_REGION or "us-east-1"
    host = AMAZON_PA_HOST or "webservices.amazon.com"
    if not (access_key and secret_key and partner_tag):
        logger.warning("Amazon PA credentials/partner tag missing - returning fallback results")
        # fallback mock results
        safe_q = quote_plus(q)[:80]
        return [
            {"title": f"{q} — Popular Choice", "price": "₹499", "url": f"https://example.com/p/{safe_q}-1", "source": "fallback"},
            {"title": f"{q} — Budget Option", "price": "₹299", "url": f"https://example.com/p/{safe_q}-2", "source": "fallback"},
            {"title": f"{q} — Premium", "price": "₹899", "url": f"https://example.com/p/{safe_q}-3", "source": "fallback"},
        ][:limit]

    # Build PA-API body
    path = "/paapi5/searchitems"
    body = {
        "PartnerType": "Associates",
        "PartnerTag": partner_tag,
        "Keywords": q,
        "SearchIndex": "All",
        "Resources": [
            "Images.Primary.Medium",
            "ItemInfo.Title",
            "ItemInfo.Features",
            "Offers.Listings.Price",
            "DetailPageURL",
            "ItemInfo.ByLineInfo",
        ],
        "ItemCount": int(limit)
    }

    try:
        status, data = _paapi_signed_post(path, body, region, host, access_key, secret_key, service="ProductAdvertisingAPI")
        logger.info("PA-API response status=%s", status)
        # parse results
        items = []
        if isinstance(data, dict):
            # PA-API places items in data.get("SearchResult", {}).get("Items", [])
            search_result = data.get("SearchResult") or {}
            raw_items = search_result.get("Items") or []
            for it in raw_items[:limit]:
                title = (it.get("ItemInfo", {}).get("Title", {}).get("DisplayValue") or "").strip()
                url = it.get("DetailPageURL") or it.get("DetailPageURL")
                price = ""
                # Offers listing price
                offers = it.get("Offers", {}).get("Listings") or []
                if offers and isinstance(offers, list):
                    price_val = offers[0].get("Price", {}).get("DisplayAmount")
                    price = price_val or ""
                thumbnail = (it.get("Images", {}).get("Primary", {}).get("Medium", {}).get("URL") or "") if it.get("Images") else ""
                items.append({"title": title, "price": price, "url": url, "thumbnail": thumbnail, "source": "amazon", "raw": it})
        # fallback: if PA returned unexpected, return fallback mocked
        if not items:
            logger.warning("PA-API returned no items; falling back")
            safe_q = quote_plus(q)[:80]
            return [
                {"title": f"{q} — Popular Choice", "price": "₹499", "url": f"https://example.com/p/{safe_q}-1", "source":"fallback"},
                {"title": f"{q} — Budget Option", "price": "₹299", "url": f"https://example.com/p/{safe_q}-2", "source":"fallback"},
            ][:limit]
        return items[:limit]
    except Exception as e:
        logger.exception("tool_search_products_pa failed: %s", e)
        safe_q = quote_plus(q)[:80]
        return [
            {"title": f"{q} — Popular Choice", "price": "₹499", "url": f"https://example.com/p/{safe_q}-1", "source":"fallback"},
            {"title": f"{q} — Budget Option", "price": "₹299", "url": f"https://example.com/p/{safe_q}-2", "source":"fallback"},
        ][:limit]

# ---------- Agent tools & runner (agentic) ----------
def tool_send_sms(to: str, body: str) -> bool:
    logger.info("tool_send_sms: to=%s body=%s", to, body[:140])
    if not twilio_client:
        logger.warning("tool_send_sms: no twilio_client configured")
        return False
    try:
        twilio_client.messages.create(body=body, from_=os.environ.get("TWILIO_NUMBER"), to=to)
        return True
    except Exception:
        logger.exception("tool_send_sms failed")
        return False

def tool_create_lead(call_sid: str, contact_number: str, notes: str) -> dict:
    lead_id = f"LEAD-{int(time.time())}"
    logger.info("tool_create_lead: %s %s", lead_id, contact_number)
    # TODO: persist to real DB
    return {"lead_id": lead_id, "contact": contact_number, "notes": notes}

def tool_transfer_to_agent(call_sid: str, phone_number: str) -> bool:
    logger.info("tool_transfer_to_agent: call=%s to=%s", call_sid, phone_number)
    # Implement Twilio transfer/dial if needed
    try:
        if not twilio_client:
            logger.warning("tool_transfer_to_agent: no twilio client")
            return False
        # Make a new <Dial> TwiML and update the call
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?><Response><Say>Transferring you now.</Say><Dial>{phone_number}</Dial></Response>"""
        return safe_update_call_with_twiml(call_sid, twiml)
    except Exception:
        logger.exception("tool_transfer_to_agent failed")
        return False

# TOOL_MAP maps tool-names to functions
TOOL_MAP = {
    "search_products": tool_search_products_pa,
    "send_sms": tool_send_sms,
    "create_lead": tool_create_lead,
    "transfer_to_agent": tool_transfer_to_agent,
}

def safe_call_tool(name: str, args: dict, call_sid: str):
    logger.info("[%s] safe_call_tool: %s %r", call_sid, name, args)
    fn = TOOL_MAP.get(name)
    if not fn:
        raise ValueError(f"unknown tool: {name}")
    try:
        return fn(**args)
    except Exception as e:
        logger.exception("[%s] tool %s failed: %s", call_sid, name, e)
        return {"error": str(e)}

def run_agent_chain(call_sid: str, transcript: str, max_rounds: int = 2, model: str | None = None) -> dict:
    """
    Agentic loop: ask LLM for plan, execute actions (tools), feed observations back, stop when final.
    """
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    rounds = 0
    messages = [
        {"role": "system", "content": (
            "You are an agent that returns JSON only. "
            "When given a user transcript, either return a 'plan' JSON with 'steps' listing tool calls to perform or a 'final' JSON with a short 'reply'. "
            "Plan format: {\"type\":\"plan\",\"steps\":[{\"kind\":\"action\",\"tool\":\"search_products\",\"args\":{...}},...]}. "
            "Final format: {\"type\":\"final\",\"reply\":\"<25-word reply>\",\"confidence\":0.8}. "
            "Do not output anything outside JSON."
        )},
        {"role": "user", "content": f"User: \"{transcript}\". Tools: {list(TOOL_MAP.keys())}. Produce plan or final."}
    ]
    log = []
    while rounds < max_rounds:
        rounds += 1
        logger.info("[%s] run_agent_chain round %d", call_sid, rounds)
        # call model (robust)
        raw = None
        try:
            # prefer new client
            try:
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAIClient()
                resp = client.chat.completions.create(model=model, messages=messages, max_tokens=400, temperature=0.0)
                raw = resp.choices[0].message.content
            except Exception as new_exc:
                logger.debug("[%s] new OpenAI client failed: %s", call_sid, new_exc)
                resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=400, temperature=0.0)
                raw = resp["choices"][0]["message"]["content"]
            logger.info("[%s] run_agent_chain raw model output len=%d", call_sid, len(raw or ""))
            log.append({"round": rounds, "raw": raw})
        except Exception as e:
            logger.exception("[%s] LLM call failed in run_agent_chain: %s", call_sid, e)
            break

        # parse JSON
        try:
            parsed = json.loads(raw)
        except Exception as e:
            logger.exception("[%s] run_agent_chain: failed to parse JSON from model: %s raw=%r", call_sid, e, raw[:400])
            # fallback reply to keep call moving
            return {"reply_text": f"I heard: {transcript}. Could you say 'books' or 'price'?", "raw_log": log}

        if parsed.get("type") == "final":
            reply = parsed.get("reply", "").strip()
            confidence = float(parsed.get("confidence", 0.5))
            return {"reply_text": reply, "confidence": confidence, "raw_log": log}

        # execute steps
        steps = parsed.get("steps", []) or []
        observations = []
        for step in steps:
            if step.get("kind") != "action":
                continue
            tool = step.get("tool")
            args = step.get("args") or {}
            obs = safe_call_tool(tool, args, call_sid)
            observations.append({"tool": tool, "args": args, "observation": obs})
            # add observation to messages for next iteration
            messages.append({"role": "assistant", "content": json.dumps({"tool": tool, "args": args})})
            messages.append({"role": "system", "content": f"OBSERVATION: {json.dumps(obs, default=str)[:1000]}"})

        messages.append({"role": "user", "content": "Observations: " + json.dumps(observations, default=str)})
        # next round

    # fallback if not final
    return {"reply_text": f"I found {len(steps)} results. Would you like links sent to your phone?", "raw_log": log}

# ---------- process_user_text -> use agentic runner ----------
def process_user_text(model_name: str, call_sid: str, user_text: str) -> dict:
    """
    Adapter to run agent chain and return the standard contract.
    """
    logger.info("[%s] process_user_text START (agentic). transcript=%r", call_sid, user_text)
    if not user_text or not str(user_text).strip():
        return {"intent": "unclear", "confidence": 0.2, "action": "ask_followup", "reply_text": "I didn't catch that. Please repeat after the beep."}
    try:
        agent_result = run_agent_chain(call_sid, user_text, max_rounds=2)
        reply = agent_result.get("reply_text", "").strip() or "Sorry, I couldn't find anything."
        conf = agent_result.get("confidence", 0.8)
        # map to simple intent/action heuristics
        intent = "info"
        action = "ask_followup"
        if "buy" in user_text.lower() or "price" in user_text.lower() or "purchase" in user_text.lower():
            intent = "purchase"
            action = "answer_with_cta"
        return {"intent": intent, "confidence": conf, "action": action, "reply_text": reply}
    except Exception as e:
        logger.exception("[%s] process_user_text agentic failed: %s", call_sid, e)
        return {"intent": "unclear", "confidence": 0.5, "action": "ask_followup", "reply_text": "Sorry, I had trouble. Could you repeat?"}

# ---------- /twiml and /recording endpoints (unchanged semantics) ----------
@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    logger.info("/twiml hit")
    resp = VoiceResponse()
    resp.say("Hello, this is our AI sales assistant. Please say something after the beep.", voice="alice")
    action_url = recording_callback_url(request)
    logger.info("Action_URL %s", action_url)
    resp.record(max_length=8, action=action_url, play_beep=True, timeout=2)
    return make_twiml_response(str(resp))

@app.post("/recording")
async def recording(request: Request, background_tasks: BackgroundTasks):
    # (keeps your robust sync+fallback scheduling logic)
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

    # minimal config
    SYNC_TIMEOUT_SECONDS = float(os.environ.get("RECORDING_SYNC_TIMEOUT", "18"))
    KEEPALIVE_SECONDS = int(os.environ.get("KEEPALIVE_SECONDS", "45"))

    loop = asyncio.get_running_loop()

    async def do_sync_processing():
        file_path = None
        transcribe_future = None
        try:
            if not recording_url:
                logger.warning("[%s] no recording_url for sync path", call_sid)
                return None
            download_url = build_download_url(recording_url)
            logger.info("[%s] sync: downloading %s", call_sid, download_url)

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
            logger.info("[%s] sync: downloaded to %s", call_sid, file_path)

            def _transcribe():
                return transcribe_with_openai(file_path)
            transcribe_future = loop.run_in_executor(None, _transcribe)
            transcript = await asyncio.wait_for(asyncio.wrap_future(asyncio.ensure_future(transcribe_future)), timeout=max(4, SYNC_TIMEOUT_SECONDS - 6))
            logger.info("[%s] sync: transcript=%r", call_sid, transcript)

            # Agent processing
            if asyncio.iscoroutinefunction(process_user_text):
                result = await process_user_text("default", call_sid, transcript)
            else:
                result = await loop.run_in_executor(None, lambda: process_user_text("default", call_sid, transcript))
            assistant_text = (result.get("reply_text") if isinstance(result, dict) else str(result)) or ""
            if not assistant_text:
                assistant_text = "Sorry, I couldn't understand that."

            s3_key = await loop.run_in_executor(None, lambda: create_and_upload_tts(assistant_text))
            logger.info("[%s] sync: uploaded tts s3://%s/%s", call_sid, S3_BUCKET, s3_key)
            tts_url = make_proxy_url(s3_key)
            full_tts_url = tts_url if tts_url.startswith("http") else (f"https://{HOSTNAME}{tts_url}" if HOSTNAME else tts_url)
            return full_tts_url
        except asyncio.TimeoutError:
            logger.warning("[%s] synchronous processing timed out", call_sid)
            return None
        except Exception as e:
            logger.exception("[%s] sync processing failed: %s", call_sid, e)
            return None
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    logger.exception("[%s] cleanup failed for %s", call_sid, file_path)

    # run sync path with timeout
    try:
        tts_play_url = await asyncio.wait_for(do_sync_processing(), timeout=SYNC_TIMEOUT_SECONDS)
    except Exception:
        tts_play_url = None

    if tts_play_url:
        logger.info("[%s] returning immediate <Play> for %s", call_sid, tts_play_url)
        record_action = recording_callback_url(request)
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{tts_play_url}</Play>
  <Record maxLength="12" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
        # schedule background update to run extra tasks if needed
        try:
            task = loop.create_task(process_recording_background(call_sid, recording_url, recording_sid, payload))
            def _bg_done(fut):
                try:
                    exc = fut.exception()
                    if exc:
                        logger.exception("[%s] background task error: %s", call_sid, exc)
                except Exception:
                    pass
            task.add_done_callback(_bg_done)
            logger.info("[%s] scheduled background task: %s", call_sid, task)
        except RuntimeError:
            background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
            logger.info("[%s] scheduled background task via BackgroundTasks fallback", call_sid)
        return make_twiml_response(twiml)

    # fallback keep-alive and schedule background processing
    try:
        task = loop.create_task(process_recording_background(call_sid, recording_url, recording_sid, payload))
        logger.info("[%s] scheduled background task (fallback): %s", call_sid, task)
    except RuntimeError:
        background_tasks.add_task(process_recording_background, call_sid, recording_url, recording_sid, payload)
        logger.info("[%s] scheduled background task via BackgroundTasks fallback", call_sid)

    twiml_keepalive = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Please hold while we process your response.</Say>
  <Pause length="{KEEPALIVE_SECONDS}"/>
</Response>"""
    return make_twiml_response(twiml_keepalive)

# ---------- background job (keeps your previous implementation, unchanged) ----------
async def process_recording_background(call_sid: str, recording_url: str | None = None, recording_sid: str | None = None, payload: dict | None = None):
    # This implementation mirrors earlier robust function: download, transcribe, call agent, TTS, update call
    file_path = None
    try:
        logger.info("[%s] process_recording_background START url=%r", call_sid, recording_url)
        download_url = build_download_url(recording_url)
        resp = None
        if download_url:
            try:
                auth = HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN) if (TWILIO_SID and TWILIO_TOKEN) else None
                resp = requests.get(download_url, auth=auth, timeout=20)
                resp.raise_for_status()
                logger.info("[%s] downloaded recording bytes=%d", call_sid, len(resp.content or b""))
            except Exception as e:
                logger.exception("[%s] failed to download recording: %s", call_sid, e)
        if resp and resp.content:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(resp.content)
            tmp.flush()
            tmp.close()
            file_path = tmp.name
            logger.info("[%s] saved recording to %s", call_sid, file_path)
            try:
                transcript = transcribe_with_openai(file_path)
                logger.info("[%s] transcription result: %r", call_sid, transcript)
            except Exception:
                logger.exception("[%s] transcription error", call_sid)
                transcript = ""
            if not transcript or not transcript.strip():
                logger.info("[%s] transcript empty; asking caller to repeat", call_sid)
                try:
                    repeat_text = "Sorry, I didn't hear that. Please repeat after the beep."
                    s3_key = create_and_upload_tts(repeat_text)
                    proxy = make_proxy_url(s3_key)
                    play_url = proxy if proxy.startswith("http") else (f"https://{HOSTNAME}{proxy}" if HOSTNAME else proxy)
                    record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"
                    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{play_url}</Play>
  <Record maxLength="12" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
                    updated = safe_update_call_with_twiml(call_sid, twiml)
                    logger.info("[%s] asked caller to repeat -> update attempted: %s", call_sid, updated)
                except Exception:
                    logger.exception("[%s] failed to ask caller to repeat", call_sid)
                safe_remove_file(file_path)
                return
            # call agent
            try:
                result = process_user_text("default", call_sid, transcript)
                if asyncio.iscoroutine(result):
                    # if process_user_text is async (unlikely here), await it
                    result = await result
                logger.info("[%s] process_user_text returned: %r", call_sid, result)
                assistant_text = result.get("reply_text") if isinstance(result, dict) else str(result)
            except Exception:
                logger.exception("[%s] process_user_text failed", call_sid)
                assistant_text = "I had trouble processing that. Could you repeat?"
        else:
            assistant_text = "Sorry, I could not hear anything that time. Please try again."

        # create tts + update call
        try:
            s3_key = create_and_upload_tts(assistant_text)
            logger.info("[%s] uploaded tts s3 key=%s", call_sid, s3_key)
            proxy = make_proxy_url(s3_key)
            play_url = proxy if proxy.startswith("http") else (f"https://{HOSTNAME}{proxy}" if HOSTNAME else proxy)
            record_action = f"https://{HOSTNAME}/recording" if HOSTNAME else "/recording"
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Play>{play_url}</Play>
  <Record maxLength="10" action="{record_action}" playBeep="true" timeout="2"/>
</Response>"""
            updated = safe_update_call_with_twiml(call_sid, twiml)
            logger.info("[%s] background update attempted -> %s", call_sid, updated)
        except Exception:
            logger.exception("[%s] TTS/upload/update failed", call_sid)
    except Exception as e:
        logger.exception("[%s] unexpected error in background: %s", call_sid, e)
    finally:
        safe_remove_file(file_path)
        logger.info("[%s] process_recording_background FINISHED", call_sid)

# ---------- tts-proxy and health ----------
@app.get("/tts-proxy")
def tts_proxy(key: str):
    if not key or ".." in key:
        return JSONResponse({"error": "invalid_key"}, status_code=400)
    try:
        meta = s3.head_object(Bucket=S3_BUCKET, Key=key)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body_stream = obj["Body"]
        content_type = meta.get("ContentType", "audio/mpeg")
        return StreamingResponse(body_stream, media_type=content_type)
    except Exception as e:
        logger.exception("tts-proxy failed for key=%s: %s", key, e)
        return JSONResponse({"error":"tts_proxy_failed", "detail": str(e)}, status_code=500)

@app.get("/")
def index():
    return PlainTextResponse("OK")
