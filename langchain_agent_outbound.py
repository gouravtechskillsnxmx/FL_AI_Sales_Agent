# langchain_agent_outbound.py
"""
Core agent logic for the Outbound AI Sales Agent.

This module intentionally does NOT create a FastAPI app (so it can be imported safely
by ws_server.py). It provides:
- SCRIPTS (in-memory)
- CONVERSATION_STATE (in-memory)
- get_script_segment_tool (robust)
- call_llm_safe (tries LangChain agent if available, with timeout)
- clean_reply_text
- process_user_text(...) -> returns dict {reply_text, used_script, next_state}
"""

import os
import re
import json
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger("fl_ai_sales.langchain")

# -------------------------
# Simple in-memory script store (can be dict or list for each script_id)
# -------------------------
SCRIPTS: Dict[str, object] = {
    "default": {
        0: "Hi, I'm calling from ExampleCorp. Do you have 30 seconds?",
        1: "Great — we help companies reduce costs by up to 20% with our automation product.",
        2: "Would you be interested in a short demo next week?",
        3: "Thanks for your time! Can I confirm your email to send details?"
    }
    # You can replace this with a vector DB retriever later.
}

# -------------------------
# In-memory conversation state (per convo_id). Persist to Redis/Postgres in prod.
# -------------------------
CONVERSATION_STATE: Dict[str, int] = {}

# -------------------------
# Utility: robust script retriever (works for dict or list scripts)
# -------------------------
def get_script_segment_tool(input_text: str) -> str:
    """
    Input formats:
      - "script_id=default;state=1"
      - "default"
    Supports SCRIPTS entries in either dict(state->line) or list/tuple form.
    """
    try:
        if ";" in input_text:
            parts = dict(part.split("=", 1) for part in input_text.split(";") if "=" in part)
            script_id = parts.get("script_id", "default")
            state = int(parts.get("state", 0))
        else:
            script_id = (input_text or "default").strip()
            state = 0
    except Exception as e:
        logger.exception("Failed parsing script input '%s': %s", input_text, e)
        script_id = "default"
        state = 0

    script = SCRIPTS.get(script_id)
    if script is None:
        logger.warning("Script id '%s' not found.", script_id)
        return "[no script found]"

    if isinstance(script, dict):
        return script.get(state, "[no more scripted lines]")

    if isinstance(script, (list, tuple)):
        if 0 <= state < len(script):
            return script[state]
        else:
            logger.info("Script '%s' index %s out of range (len=%d).", script_id, state, len(script))
            return "[no more scripted lines]"

    logger.warning("Script '%s' has unsupported type %s", script_id, type(script))
    return "[invalid script format]"

# -------------------------
# Cleaning helper for model outputs
# -------------------------
def clean_reply_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Remove surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # Trim whitespace/newlines
    s = re.sub(r'^\s+|\s+$', '', s)
    return s

# -------------------------
# LangChain agent / LLM initialization (optional & best-effort)
# - If LangChain isn't installed or OPENAI_API_KEY missing, agent remains None
# -------------------------
agent = None
llm = None
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        # Attempt to initialize a LangChain chat model + agent with our tool
        from langchain.chat_models import ChatOpenAI
        from langchain.tools import Tool
        from langchain.agents import initialize_agent, AgentType

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(temperature=0.2, model_name=model_name)
        # register local tool that returns script segments
        tools = [
            Tool(name="get_script_segment", func=get_script_segment_tool,
                 description="Returns the next scripted segment given script_id and state. Input 'script_id=default;state=1'")
        ]
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
        logger.info("LangChain agent initialized with model %s", model_name)
    else:
        logger.info("OPENAI_API_KEY not set; skipping LangChain agent initialization")
except Exception as e:
    agent = None
    llm = None
    logger.warning("LangChain initialization failed or not available: %s", e)

# -------------------------
# Safe LLM call wrapper
# -------------------------
def call_llm_safe(prompt: str, timeout_seconds: int = 8) -> Optional[str]:
    """
    Try to call the LangChain agent (if configured) with a timeout.
    Returns the agent result string or None if agent isn't available or fails.
    """
    if agent is None:
        logger.debug("call_llm_safe: no agent configured")
        return None

    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(agent.run, prompt)
            result = future.result(timeout=timeout_seconds)
        return result
    except concurrent.futures.TimeoutError:
        logger.warning("call_llm_safe: agent timed out after %s seconds", timeout_seconds)
        return None
    except Exception as e:
        logger.exception("call_llm_safe: agent call failed: %s", e)
        return None

# -------------------------
# Main business logic (used by ws_server)
# -------------------------
def process_user_text(script_id: str, convo_id: str, user_text: str) -> Dict:
    """
    Decide scripted vs dynamic reply.
    Returns: {"reply_text": str, "used_script": bool, "next_state": int}
    """
    script_id = script_id or "default"
    convo_id = convo_id or "global"
    user_text = (user_text or "").strip()

    state = CONVERSATION_STATE.get(convo_id, 0)
    script_input = f"script_id={script_id};state={state}"
    scripted_line = get_script_segment_tool(script_input)

    # Heuristic to detect dynamic responses (questions / pricing keywords)
    lower = user_text.lower()
    question_triggers = ["?", "price", "cost", "how much", "pricing", "demo", "interested", "no thanks", "not interested"]
    is_dynamic = any(t in lower for t in question_triggers) or user_text.endswith("?")

    if is_dynamic:
        # Controlled prompt asking for a short reply
        prompt = (
            "You are a concise outbound sales assistant for ExampleCorp. Keep replies under 30 words, "
            "professional and compliant with company voice.\n"
            f"CONTEXT: Current scripted line: '{scripted_line}'\nUSER: '{user_text}'\n"
            "Provide a short reply only (no extra commentary)."
        )
        start = time.time()
        agent_result = call_llm_safe(prompt, timeout_seconds=8)
        duration = time.time() - start
        logger.info("LLM call for convo=%s took %.2fs; result present=%s", convo_id, duration, bool(agent_result))
        if agent_result:
            # Try to parse structured JSON if model returned one
            cleaned = clean_reply_text(agent_result)
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except Exception:
                parsed = None

            if isinstance(parsed, dict) and "reply" in parsed:
                reply = clean_reply_text(parsed.get("reply", ""))
                advance = bool(parsed.get("advance_state", False))
            else:
                reply = cleaned
                advance = False

            used_script = False
            if advance:
                CONVERSATION_STATE[convo_id] = state + 1
        else:
            # fallback to scripted line
            logger.info("LLM unavailable or failed; falling back to scripted line for convo=%s", convo_id)
            reply = scripted_line
            used_script = True
            CONVERSATION_STATE[convo_id] = state + 1

    else:
        # Use scripted reply and advance
        reply = scripted_line
        used_script = True
        CONVERSATION_STATE[convo_id] = state + 1

    reply = clean_reply_text(reply)
    return {"reply_text": reply, "used_script": used_script, "next_state": CONVERSATION_STATE.get(convo_id, state)}

# End of langchain_agent_outbound.py
