"""
Sentiment Analyzer
Sub-task 2: Sentiment Analysis using fine-tuned DistilBERT (SST-2).
Analyzes sentiment of IT ticket text to understand user frustration level.
"""

from transformers import pipeline

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

PRIORITY_MAPPING = {
    "NEGATIVE": {
        "priority": "HIGH",
        "color": "red",
        "icon": "🔴",
        "action": "Escalate immediately — user is likely frustrated.",
    },
    "POSITIVE": {
        "priority": "NORMAL",
        "color": "green",
        "icon": "🟢",
        "action": "Standard SLA applies.",
    },
}


def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def analyze_sentiment(text: str, analyzer) -> dict:
    """Analyze sentiment and map to ticket priority."""
    result = analyzer(text[:512])[0]
    label = result["label"]
    score = result["score"]

    # Determine urgency level based on confidence
    if label == "NEGATIVE":
        if score > 0.90:
            urgency = "Critical"
        elif score > 0.75:
            urgency = "High"
        else:
            urgency = "Medium"
    else:
        urgency = "Low"

    info = PRIORITY_MAPPING.get(label, PRIORITY_MAPPING["POSITIVE"])
    return {
        "sentiment": label,
        "confidence": round(score * 100, 1),
        "priority": info["priority"],
        "urgency": urgency,
        "color": info["color"],
        "icon": info["icon"],
        "action": info["action"],
        "positive_score": round((1 - score) * 100, 1) if label == "NEGATIVE" else round(score * 100, 1),
        "negative_score": round(score * 100, 1) if label == "NEGATIVE" else round((1 - score) * 100, 1),
    }
