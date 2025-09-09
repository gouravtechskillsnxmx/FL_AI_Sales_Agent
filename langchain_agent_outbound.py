"""
LangChain Agent example for an Outbound AI Sales Agent (single-file PoC)

Features:
- Exposes two LangChain tools: get_script_segment and call_llm
- Builds a LangChain agent that decides whether to use a scripted reply or generate a dynamic reply
- Provides a minimal FastAPI HTTP endpoint (/respond) for testing

Instructions (quick):
1. Set environment variables:
   - OPENAI_API_KEY (your OpenAI API key)
2. Install dependencies (recommended in a venv):
   pip install -r requirements.txt

requirements.txt (example):
fastapi
uvicorn[standard]
langchain
openai
pydantic
python-multipart

Start locally for quick testing:
    uvicorn langchain_agent_outbound:app --host 0.0.0.0 --port 8000

Endpoint:
POST /respond
JSON body: {"script_id": "default", "user_text": "Hello, tell me about pricing"}

Render deploy notes:
- Create a Git repo containing this file and a requirements.txt.
- On Render (Web Service), set the Start Command to:
    gunicorn -k uvicorn.workers.UvicornWorker langchain_agent_outbound:app --bind 0.0.0.0:$PORT
- Add environment variable OPENAI_API_KEY in Render Dashboard.

Caveats & Next steps:
- This is a PoC. Add authentication, rate-limiting, conversation persistence, and proper prompt engineering for production.
- Replace in-memory script store with a DB or vector DB for RAG.

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Dict, Optional
import logging

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------
# Simple in-memory script store
# --------------------
SCRIPTS: Dict[str, Dict[int, str]] = {
    "default": {
        0: "Hi, I'm calling from ExampleCorp. Do you have 30 seconds?",
        1: "Great — we help companies reduce costs by up to 20% with our automation product.",
        2: "Would you be interested in a short demo next week?",
        3: "Thanks for your time! Can I confirm your email to send details?"
    }
}

# Simple state tracker per 'conversation' (in-memory) — production: replace with persistent store
CONVERSATION_STATE: Dict[str, int] = {}

# --------------------
# LangChain LLM and tools
# --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. The call_llm tool will fail until the key is provided.")

llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")  # choose model you have access to

# Tool: get_script_segment
def get_script_segment_tool(input_text: str) -> str:
    """Expects input like: "script_id=default;state=1" or just script_id.
    Returns the next scripted line based on state.
    """
    try:
        # parse input
        parts = {k: v for part in input_text.split(";") for k, v in [part.split("=")]} if ";" in input_text else {"script_id": input_text}
    except Exception:
        parts = {"script_id": input_text}

    script_id = parts.get("script_id", "default")
    state = int(parts.get("state", 0))
    script = SCRIPTS.get(script_id, {})
    # Return the line at state if exists, else return a default suggestion
    return script.get(state, "[no more scripted lines]")

# Tool: call_llm
def call_llm_tool(prompt: str) -> str:
    """Sends prompt to the LLM and returns the text response."""
    # Use a short, controlled temperature for predictable replies
    try:
        resp = llm.generate([{"role": "user", "content": prompt}])
        # langchain ChatOpenAI.generate returns a Generation object; extract text
        outputs = []
        for g in resp.generations:
            if g and len(g) > 0:
                outputs.append(g[0].text)
        return "\n---\n".join(outputs) if outputs else ""
    except Exception as e:
        logger.exception("LLM call failed")
        return f"[llm_error] {str(e)}"

# Create Tool wrappers for LangChain
tools = [
    Tool(name="get_script_segment", func=get_script_segment_tool, description="Returns the next scripted segment given a script id and state. Input format: 'script_id=default;state=1'"),
    Tool(name="call_llm", func=call_llm_tool, description="Call the LLM with a prompt and get a text reply.")
]

# Memory for agent (short-term). In production, attach conversation-specific memory.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
)

# --------------------
# FastAPI app & testing endpoint
# --------------------
app = FastAPI(title="LangChain Outbound Agent PoC")

class RespondRequest(BaseModel):
    script_id: Optional[str] = "default"
    convo_id: Optional[str] = None
    user_text: str

class RespondResponse(BaseModel):
    reply_text: str
    used_script: bool
    next_state: int

@app.post("/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest):
    """Decide scripted vs dynamic reply.

    Strategy (simple):
    1. Fetch current state for convo_id (default 0)
    2. Get the scripted line for that state
    3. If user_text matches a small set of triggers (e.g., yes/no questions, or contains keyword), use scripted flow and advance state
    4. Otherwise, call agent (which may call call_llm tool) to generate a dynamic reply
    """
    script_id = payload.script_id or "default"
    convo_id = payload.convo_id or "global"
    user_text = payload.user_text.strip()

    # current state
    state = CONVERSATION_STATE.get(convo_id, 0)

    # get scripted line
    script_input = f"script_id={script_id};state={state}"
    scripted_line = get_script_segment_tool(script_input)

    # Very small heuristic to decide: if user asked a question (ends with ?) or contains keyword 'price' or 'cost', go dynamic
    lower = user_text.lower()
    question_triggers = ["?", "price", "cost", "how much", "pricing", "demo", "interested", "no thanks", "not interested"]
    is_dynamic = any(t in lower for t in question_triggers) or (user_text.endswith("?"))

    if is_dynamic:
        # Build a controlled prompt which includes the current script context and instructs LLM to be brief, brand-safe
        prompt = (
            f"You are an outbound sales assistant for ExampleCorp. Keep replies under 30 words, professional and concise."
            f"\nCONTEXT: Current scripted line: '{scripted_line}'\nUSER: '{user_text}'\nACTION: Provide a short reply consistent with the script and ask one follow-up question if appropriate."
        )
        # Use the agent (which can call call_llm tool)
        try:
            logger.info("Calling agent for dynamic reply")
            agent_result = agent.run(prompt)
            reply = agent_result.strip()
            used_script = False
        except Exception as e:
            logger.exception("Agent failed, falling back to scripted reply")
            reply = scripted_line
            used_script = True
    else:
        # Use scripted reply and advance the state
        reply = scripted_line
        used_script = True
        CONVERSATION_STATE[convo_id] = state + 1

    return RespondResponse(reply_text=reply, used_script=used_script, next_state=CONVERSATION_STATE.get(convo_id, state))

# Simple health check
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
