"""
RAG Pipeline — Retrieval-Augmented Generation for IT Knowledge Base Chatbot.

Architecture:
  - Loader: reads .txt files from knowledge_base/
  - Chunker: splits documents into overlapping chunks
  - Embedder: sentence-transformers/all-MiniLM-L6-v2
  - Vector Store: FAISS (in-memory, rebuilt per session)
  - Reader: distilbert-base-cased-distilled-squad (extractive QA on retrieved chunks)
"""

import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

KB_DIR = Path(__file__).parent / "knowledge_base"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-cased-distilled-squad"

CHUNK_SIZE = 300       # words per chunk
CHUNK_OVERLAP = 50     # words overlapping between consecutive chunks
TOP_K = 3              # number of chunks to retrieve


def load_documents() -> List[dict]:
    """Load all .txt files from the knowledge base directory."""
    docs = []
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "content": text})
    return docs


def chunk_documents(docs: List[dict]) -> List[dict]:
    """Split documents into overlapping word-level chunks."""
    chunks = []
    for doc in docs:
        words = doc["content"].split()
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"source": doc["source"], "text": chunk_text})
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index(chunks: List[dict], embedder: SentenceTransformer) -> Tuple[faiss.Index, List[dict]]:
    """Embed chunks and build a FAISS flat L2 index."""
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks


def retrieve(query: str, index: faiss.Index, chunks: List[dict],
             embedder: SentenceTransformer, top_k: int = TOP_K) -> List[dict]:
    """Retrieve the top-k most relevant chunks for a query."""
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(dist),
            })
    return results


def answer_question(query: str, context: str, qa_pipeline) -> dict:
    """Run extractive QA on the concatenated retrieved context."""
    result = qa_pipeline(question=query, context=context[:3000])
    return {
        "answer": result["answer"],
        "confidence": round(result["score"] * 100, 1),
        "start": result["start"],
        "end": result["end"],
    }


class RAGPipeline:
    """End-to-end RAG pipeline: retrieval + extractive question answering."""

    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.qa = pipeline("question-answering", model=QA_MODEL)
        self._build()

    def _build(self):
        docs = load_documents()
        self.chunks = chunk_documents(docs)
        self.index, self.chunks = build_index(self.chunks, self.embedder)

    def query(self, question: str) -> dict:
        """Retrieve relevant chunks and answer the question."""
        retrieved = retrieve(question, self.index, self.chunks, self.embedder, TOP_K)
        if not retrieved:
            return {
                "answer": "I could not find relevant information in the knowledge base.",
                "confidence": 0.0,
                "sources": [],
                "context_used": "",
            }

        # Concatenate retrieved chunks as context (deduplicated)
        seen = set()
        context_parts = []
        for r in retrieved:
            if r["text"] not in seen:
                context_parts.append(r["text"])
                seen.add(r["text"])
        context = "\n\n".join(context_parts)

        qa_result = answer_question(question, context, self.qa)

        return {
            "answer": qa_result["answer"],
            "confidence": qa_result["confidence"],
            "sources": [r["source"] for r in retrieved],
            "context_used": context,
            "retrieved_chunks": retrieved,
        }

    def rebuild_index(self):
        """Reload documents and rebuild the FAISS index."""
        self._build()
