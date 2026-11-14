import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EMBED_ID, LLM_ID
from .engine import NBAEngine

engine = NBAEngine()


async def initialize():
    try:
        await asyncio.to_thread(engine.load)
    except Exception:
        logging.exception("Model initialization failed")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(initialize())
    yield
    await task


app = FastAPI(title="CourtSide · Local NBA RAG", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=600)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"ready": engine.ready, "phase": engine.phase, "error": engine.error,
            "model": LLM_ID, "embedding_model": EMBED_ID,
            "chunks": len(engine.chunks),
            "documents": [{k: d[k] for k in ("title", "url", "fetched_at")} for d in engine.documents]}


@app.post("/api/chat")
async def chat(body: Question):
    question = body.question.strip()
    if len(question) < 3:
        raise HTTPException(422, "Please enter a question of at least 3 characters.")
    if not engine.ready:
        raise HTTPException(503, engine.error or engine.phase)
    try:
        return await asyncio.to_thread(engine.ask, question)
    except Exception:
        logging.exception("Generation failed")
        raise HTTPException(500, "Could not generate an answer. Check the server log and try again.")
