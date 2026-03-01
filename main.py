"""
FastAPI Backend for Medical RAG Assistant
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_pipeline import pipeline


# ---------------------------------------------------------------------------
# Lifespan: initialize the RAG pipeline on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the server starts."""
    print("Starting Medical RAG Assistant...")
    pipeline.initialize()
    yield
    print("Shutting down Medical RAG Assistant...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Medical RAG Assistant",
    description="AI-powered medical question answering using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    max_tokens: int = 1024
    temperature: float = 0.0


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    response_time: float


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Process a medical question through the RAG pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start_time = time.time()

    result = pipeline.query(
        user_input=request.question,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    elapsed = round(time.time() - start_time, 2)

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        response_time=elapsed,
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline._initialized,
    }
