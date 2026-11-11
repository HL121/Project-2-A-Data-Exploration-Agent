import asyncio
import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from context_panel import AgentContexts, PanelHooks, add_context_panel
from scripts.lead import deep_research


HERE = Path(__file__).parent
sessions: dict[str, AgentContexts] = {}
# /chat conversations: session_id -> completed turns, used to answer a clarifying question.
chat_sessions: dict[str, list[dict[str, str]]] = {}

app = FastAPI()
add_context_panel(app, sessions)


class ToolCallRecorder(PanelHooks):
    """Panel hooks that also keep every agent's tool calls, in the Project 1 shape."""

    def __init__(self, contexts: AgentContexts, emit):
        super().__init__(contexts, emit)
        self.tool_calls: list[dict] = []

    async def on_tool_end(self, context, agent, tool, result):
        raw_args = getattr(context, "tool_arguments", None)
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (TypeError, ValueError):
            args = raw_args
        self.tool_calls.append({
            "agent": agent.name,
            "name": tool.name,
            "args": args,
            "result": json.loads(json.dumps(result, default=str)),
        })


def question_with_history(turns: list[dict[str, str]], message: str) -> str:
    """Fold earlier turns into the question, as the web page does, so an answer
    to a clarifying question reaches the Lead with the original request."""
    if not turns:
        return message
    lines = ["Earlier in this conversation:"]
    for turn in turns:
        if turn["role"] == "user":
            lines.append("User: " + turn["text"])
        elif turn["kind"] == "clarification":
            lines.append("CardWise asked: " + turn["text"][:1500])
        else:
            lines.append("CardWise delivered a research report for the request above.")
    lines += ["", "Latest user message: " + message]
    return "\n".join(lines)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer one message and return every agent's tool calls, each with an agent field."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    session_id = request.session_id or str(uuid.uuid4())
    turns = chat_sessions.setdefault(session_id, [])
    recorder = ToolCallRecorder(AgentContexts(), lambda event: None)
    try:
        response = await deep_research(question_with_history(turns, request.message), recorder)
    except Exception as exc:
        # Auth, quota or model failures: report them in the response, not as a 500.
        response = f"Research failed: {type(exc).__name__}: {str(exc)[:300]}"
        return ChatResponse(response=response, session_id=session_id, tool_calls=recorder.tool_calls)
    # Without any Researcher tool call, the Lead asked for more information.
    researched = any(call["agent"] != "Lead" for call in recorder.tool_calls)
    turns += [
        {"role": "user", "text": request.message},
        {"role": "assistant", "kind": "report" if researched else "clarification", "text": response},
    ]
    return ChatResponse(response=response, session_id=session_id, tool_calls=recorder.tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    chat_sessions.pop(session_id, None)
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/run")
async def run(question: str):
    """Research one credit-card question and stream progress as Server-Sent Events."""
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    queue: asyncio.Queue = asyncio.Queue()
    hooks = PanelHooks(sessions[run_id], queue.put_nowait)

    async def work():
        try:
            report = await deep_research(question, hooks)
            queue.put_nowait({"type": "done", "markdown": report})
        except Exception as exc:
            queue.put_nowait({"type": "error", "message": f"{type(exc).__name__}: {str(exc)[:500]}"})

    task = asyncio.create_task(work())

    async def events():
        try:
            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event["type"] in ("done", "error"):
                    break
        finally:
            task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
