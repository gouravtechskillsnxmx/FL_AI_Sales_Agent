# langchain_agent_outbound.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, logging
from typing import Dict, Optional

# existing imports for langchain, agent, memory etc.
# from langchain.chat_models import ChatOpenAI
# ... your current LLM/agent setup ...
# --- Add / ensure these definitions are present near the top of langchain_agent_outbound.py ---

from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Simple in-memory scripted store (edit to your real script content)
SCRIPTS: Dict[str, Dict[int, str]] = {
    "default": {
        0: "Hi, I'm calling from ExampleCorp. Do you have 30 seconds?",
        1: "Great — we help companies reduce costs by up to 20% with our automation product.",
        2: "Would you be interested in a short demo next week?",
        3: "Thanks for your time! Can I confirm your email to send details?"
    }
}

# conversation state store (in-memory). Persist in Redis/DB for production.
CONVERSATION_STATE: Dict[str, int] = {}

def get_script_segment_tool(input_text: str) -> str:
    """
    Minimal replacement for the LangChain tool. Accepts either:
      - "script_id=default;state=1"
      - or simply "default" (in which case state defaults to 0)
    Returns the script line or a fallback string.
    """
    try:
        if ";" in input_text:
            parts = dict(part.split("=", 1) for part in input_text.split(";") if "=" in part)
            script_id = parts.get("script_id", "default")
            state = int(parts.get("state", 0))
        else:
            script_id = input_text or "default"
            state = 0
    except Exception:
        script_id = "default"
        state = 0

    script = SCRIPTS.get(script_id, {})
    return script.get(state, "[no more scripted lines]")

# Optional: lightweight wrapper for calling LLM (if you already configured `agent` or `llm`,
# you can keep your original call_llm logic). Here we add a safe fallback.
def call_llm_safe(prompt: str, timeout_seconds: int = 8) -> str:
    """
    Try to call agent.run(prompt) if `agent` exists in globals.
    Otherwise, return a fallback short reply.
    """
    if "agent" in globals() and globals().get("agent") is not None:
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(globals()["agent"].run, prompt)
                return future.result(timeout=timeout_seconds)
        except Exception as e:
            logger.exception("call_llm_safe: agent call failed: %s", e)
            return "[llm_error]"
    else:
        # No agent configured — fallback
        logger.info("call_llm_safe: no agent found, falling back to scripted reply")
        return "[service_unavailable]"

logger = logging.getLogger(__name__)






# keep your SCRIPTS, CONVERSATION_STATE, agent, memory, llm etc. unchanged above

class RespondRequest(BaseModel):
    script_id: Optional[str] = "default"
    convo_id: Optional[str] = None
    user_text: str

class RespondResponse(BaseModel):
    reply_text: str
    used_script: bool
    next_state: int


# langchain_agent_outbound.py

# Stores conversation state per conversation ID
CONVERSATION_STATE: dict = {}

# Example script storage (if not already defined)
SCRIPTS = {
    "default": [
        "Hi, I'm calling from ExampleCorp. We help companies reduce costs by up to 20% with our automation product. What specific services are you interested in?",
        "Our solution integrates easily with existing systems. Would you like me to send over more details or schedule a quick demo?",
        "Great, thank you for your time. We'll follow up shortly with more information."
    ]
}


# ---------- New reusable function ----------
def process_user_text(script_id: str, convo_id: str, user_text: str) -> Dict:
    """
    Core business logic previously in /respond endpoint.
    Returns a dict: {'reply_text': str, 'used_script': bool, 'next_state': int}
    """
    script_id = script_id or "default"
    convo_id = convo_id or "global"
    user_text = user_text.strip()

    state = CONVERSATION_STATE.get(convo_id, 0)
    script_input = f"script_id={script_id};state={state}"
    scripted_line = get_script_segment_tool(script_input)

    # simple heuristic to decide dynamic vs scripted (keep your existing rules)
    lower = user_text.lower()
    question_triggers = ["?", "price", "cost", "how much", "pricing", "demo", "interested", "no thanks", "not interested"]
    is_dynamic = any(t in lower for t in question_triggers) or user_text.endswith("?")

    if is_dynamic:
        prompt = (
            f"You are an outbound sales assistant for ExampleCorp. Keep replies under 30 words, "
            f"professional and concise.\nCONTEXT: Current scripted line: '{scripted_line}'\nUSER: '{user_text}'\n"
        )
        # Use the same threadpool / timeout wrapper you have in your app to avoid blocking
        import concurrent.futures
        LLM_TIMEOUT_SECONDS = 8
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(agent.run, prompt)
                agent_result = future.result(timeout=LLM_TIMEOUT_SECONDS)
            reply = str(agent_result).strip()
            used_script = False
        except concurrent.futures.TimeoutError:
            logger.warning("LLM timed out — falling back to scripted reply")
            reply = scripted_line
            used_script = True
            # advance script since fallback uses scripted flow
            CONVERSATION_STATE[convo_id] = state + 1
        except Exception as e:
            logger.exception("Agent failed, falling back to script: %s", e)
            reply = scripted_line
            used_script = True
            CONVERSATION_STATE[convo_id] = state + 1
    else:
        # Use scripted reply and advance state
        reply = scripted_line
        used_script = True
        CONVERSATION_STATE[convo_id] = state + 1

    # Clean the reply (strip surrounding quotes/newlines)
    def clean_reply_text(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return s

    reply = clean_reply_text(reply)
    return {"reply_text": reply, "used_script": used_script, "next_state": CONVERSATION_STATE.get(convo_id, state)}

# ---------- existing /respond endpoint now delegates to process_user_text ----------
app = FastAPI()

@app.post("/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest):
    result = process_user_text(payload.script_id, payload.convo_id or "global", payload.user_text)
    return RespondResponse(reply_text=result["reply_text"], used_script=result["used_script"], next_state=result["next_state"])

# health endpoint as before
@app.get("/health")
async def health():
    return {"status": "ok"}
