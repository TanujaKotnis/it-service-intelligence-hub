"""
Text Summarizer
Sub-task 3: Text Summarization using DistilBART.
Generates concise summaries of long IT ticket descriptions for technician handoffs.
"""

from transformers import pipeline

SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"


def load_summarizer():
    return pipeline(
        "summarization",
        model=SUMMARIZATION_MODEL,
        min_length=30,
        max_length=150,
    )


def summarize_text(text: str, summarizer, max_length: int = 150, min_length: int = 30) -> dict:
    """Summarize a long IT ticket description."""
    if len(text.split()) < 40:
        return {
            "summary": text,
            "original_length": len(text.split()),
            "summary_length": len(text.split()),
            "compression_ratio": 100.0,
            "skipped": True,
            "reason": "Text is already concise (fewer than 40 words).",
        }

    # Truncate to model max token limit
    truncated = " ".join(text.split()[:900])
    result = summarizer(
        truncated,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    summary = result[0]["summary_text"]
    orig_words = len(text.split())
    summ_words = len(summary.split())

    return {
        "summary": summary,
        "original_length": orig_words,
        "summary_length": summ_words,
        "compression_ratio": round((1 - summ_words / orig_words) * 100, 1),
        "skipped": False,
        "reason": "",
    }
