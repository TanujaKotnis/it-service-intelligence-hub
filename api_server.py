"""
IT Service Intelligence Hub — REST API Server
Exposes the same NLP models as the Streamlit app via FastAPI endpoints.

Run locally:
    uvicorn api_server:app --reload --port 8000

Then open: http://127.0.0.1:8000/docs  (auto-generated Swagger UI)
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.classifier import load_classifier, classify_ticket
from models.sentiment_analyzer import load_sentiment_analyzer, analyze_sentiment
from models.summarizer import load_summarizer, summarize_text
from rag.rag_pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Shared model state — loaded once at startup
# ---------------------------------------------------------------------------

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models…")
    _state["classifier"], _state["clf_mode"] = load_classifier()
    _state["sentiment"] = load_sentiment_analyzer()
    _state["summarizer"] = load_summarizer()
    _state["rag"] = RAGPipeline()
    print("All models ready.")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IT Service Intelligence Hub API",
    description=(
        "REST API for AI-powered IT helpdesk automation. "
        "Provides ticket classification, sentiment analysis, "
        "text summarization, and RAG-based knowledge base Q&A."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def ok(data: dict, message: str = "success") -> dict:
    return {"success": True, "data": data, "message": message}


def err(message: str, status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"success": False, "data": None, "message": message},
    )


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    question: str

class AnalyzeRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Service health check — returns model status and available endpoints."""
    return ok({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models": {
            "classifier": _state.get("clf_mode", "not_loaded"),
            "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
            "summarizer": "sshleifer/distilbart-cnn-12-6",
            "rag_embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "rag_reader": "distilbert-base-cased-distilled-squad",
        },
        "endpoints": [
            "GET  /api/health",
            "POST /api/classify",
            "POST /api/sentiment",
            "POST /api/summarize",
            "POST /api/rag/ask",
            "POST /api/analyze",
        ],
    })


@app.post("/api/classify")
def classify(req: TextRequest):
    """Classify an IT support ticket into one of 7 categories."""
    if not req.text.strip():
        return err("Field 'text' must not be empty.")
    result = classify_ticket(req.text, _state["classifier"], _state["clf_mode"])
    return ok({
        "category": result["category"],
        "display_name": result["display_name"],
        "confidence": result["confidence"],
        "description": result["description"],
        "mode": result["mode"],
        "all_scores": result["all_scores"],
    })


@app.post("/api/sentiment")
def sentiment(req: TextRequest):
    """Analyse sentiment of an IT ticket and return priority recommendation."""
    if not req.text.strip():
        return err("Field 'text' must not be empty.")
    result = analyze_sentiment(req.text, _state["sentiment"])
    return ok({
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "priority": result["priority"],
        "urgency": result["urgency"],
        "action": result["action"],
        "scores": {
            "positive": result["positive_score"],
            "negative": result["negative_score"],
        },
    })


@app.post("/api/summarize")
def summarize(req: TextRequest):
    """Summarize a long IT ticket description into a concise technician handoff note."""
    if not req.text.strip():
        return err("Field 'text' must not be empty.")
    result = summarize_text(req.text, _state["summarizer"])
    return ok({
        "summary": result["summary"],
        "original_word_count": result["original_length"],
        "summary_word_count": result["summary_length"],
        "compression_ratio_pct": result["compression_ratio"],
        "skipped": result["skipped"],
        "skip_reason": result.get("reason", ""),
    })


@app.post("/api/rag/ask")
def rag_ask(req: QuestionRequest):
    """Answer an IT support question using the RAG knowledge base."""
    if not req.question.strip():
        return err("Field 'question' must not be empty.")
    result = _state["rag"].query(req.question)
    answer = result["answer"]
    if result["confidence"] < 1:
        answer = (
            "I couldn't find a precise answer in the knowledge base. "
            "Please contact the IT Helpdesk directly at 0800-ACME-IT "
            "or visit https://itsupport.acme.com."
        )
    return ok({
        "answer": answer,
        "confidence_pct": result["confidence"],
        "sources": list(set(result["sources"])),
    })


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Full workflow: classify + sentiment + summarize a ticket in one call."""
    if not req.text.strip():
        return err("Field 'text' must not be empty.")
    clf = classify_ticket(req.text, _state["classifier"], _state["clf_mode"])
    sent = analyze_sentiment(req.text, _state["sentiment"])
    summ = summarize_text(req.text, _state["summarizer"])
    return ok({
        "classification": {
            "category": clf["category"],
            "display_name": clf["display_name"],
            "confidence": clf["confidence"],
        },
        "sentiment": {
            "label": sent["sentiment"],
            "confidence": sent["confidence"],
            "urgency": sent["urgency"],
            "priority": sent["priority"],
            "action": sent["action"],
        },
        "summary": summ["summary"],
        "compression_ratio_pct": summ["compression_ratio"],
    })
