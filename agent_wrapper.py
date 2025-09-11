# agent_wrapper.py
import asyncio
from langchain_agent_outbound import OutboundAgent  # adjust import if needed

# instantiate once (load prompts, DB, etc.)
_agent = OutboundAgent()  # pass config/db clients if your class needs them

async def process_utterance(call_sid: str, text: str) -> dict:
    """
    Async wrapper used by ws_server.
    Returns: {"text": "<reply>", "action": "play"|"script"|"hangup", "meta": {...}}
    """
    # If OutboundAgent has a synchronous method, run it in executor to avoid blocking
    loop = asyncio.get_running_loop()
    if asyncio.iscoroutinefunction(_agent.process_utterance):
        result = await _agent.process_utterance(call_sid, text)
    else:
        result = await loop.run_in_executor(None, lambda: _agent.process_utterance(call_sid, text))
    # Normalize result
    return {
        "text": result.get("text") if isinstance(result, dict) else str(result),
        "action": result.get("action", "play") if isinstance(result, dict) else "play",
        "meta": result.get("meta", {}) if isinstance(result, dict) else {}
    }
