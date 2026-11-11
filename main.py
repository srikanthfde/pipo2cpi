"""PIPO2CPI API: endpoint routing only. All agent logic lives in agent.py.

Run:  uvicorn main:app --reload   then open http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import agent

app = FastAPI(title="PIPO2CPI", version="1.0.0")
FRONTEND = Path(__file__).parent / "frontend" / "src" / "App.jsx"

# Minimal shell so the single App.jsx runs without a build step (React + Babel from CDN).
INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PIPO2CPI</title>
<script crossorigin src="https://unpkg.com/react@18.3.1/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone@7/babel.min.js"></script></head>
<body><div id="root"></div><script type="text/babel" data-presets="react" src="/App.jsx"></script></body></html>"""


class StageRequest(BaseModel):
    stage: str
    ico: dict = {}
    prior: dict = {}
    documents: list[str] = []


class ChatRequest(BaseModel):
    message: str
    agent: str = "orchestrator"
    history: list[dict] = []


class EvalRequest(BaseModel):
    include_llm: bool = False


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except agent.GuardrailViolation as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (agent.AgentConfigError, agent.PipoError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model call failed: {type(exc).__name__}: {str(exc)[:300]}")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/App.jsx")
def frontend():
    return FileResponse(FRONTEND, media_type="text/javascript")


@app.get("/api/health")
def health():
    return agent.health()


@app.get("/api/agents")
def agents():
    return agent.describe()


@app.get("/api/scenarios")
def scenarios():
    return _call(agent.fetch_scenarios)


@app.post("/api/stage")
def stage(req: StageRequest):
    return _call(agent.run_stage, req.stage, req.ico, req.prior, req.documents)


@app.post("/api/chat")
def chat(req: ChatRequest):
    return _call(agent.run_chat, req.agent, req.message, req.history)


@app.post("/api/eval")
def evaluate(req: EvalRequest):
    return _call(agent.run_evals, req.include_llm)
