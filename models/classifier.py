"""
IT Ticket Classifier
Sub-task 1: Text Classification using zero-shot or fine-tuned DistilBERT.
Falls back to zero-shot if the fine-tuned model is not found.
"""

import os
from pathlib import Path
from transformers import pipeline

FINE_TUNED_PATH = Path(__file__).parent.parent / "fine_tuning" / "saved_model"

TICKET_CATEGORIES = [
    "network_connectivity",
    "software_installation",
    "hardware_malfunction",
    "security_incident",
    "account_access",
    "email_issues",
    "printer_problems",
]

CATEGORY_DISPLAY = {
    "network_connectivity": "🌐 Network Connectivity",
    "software_installation": "💿 Software Installation",
    "hardware_malfunction": "🖥️ Hardware Malfunction",
    "security_incident": "🔒 Security Incident",
    "account_access": "🔑 Account Access",
    "email_issues": "📧 Email Issues",
    "printer_problems": "🖨️ Printer Problems",
}

CATEGORY_INFO = {
    "network_connectivity": "Issues related to internet, VPN, LAN, or Wi-Fi connectivity.",
    "software_installation": "Problems installing, updating, or uninstalling software applications.",
    "hardware_malfunction": "Physical device failures — laptop, monitor, keyboard, docking station.",
    "security_incident": "Phishing, malware, unauthorized access, or data breach reports.",
    "account_access": "Password resets, locked accounts, MFA issues, or SSO problems.",
    "email_issues": "Outlook, email sending/receiving, calendar sync, or Teams issues.",
    "printer_problems": "Printer offline, paper jam, driver issues, or print quality problems.",
}


def load_classifier():
    """Load fine-tuned model if available, otherwise use zero-shot classification."""
    if FINE_TUNED_PATH.exists():
        print(f"Loading fine-tuned classifier from {FINE_TUNED_PATH}")
        return pipeline(
            "text-classification",
            model=str(FINE_TUNED_PATH),
            return_all_scores=True,
        ), "fine_tuned"
    else:
        print("Fine-tuned model not found. Using zero-shot classification.")
        return pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
        ), "zero_shot"


def classify_ticket(text: str, classifier, mode: str) -> dict:
    """Classify an IT helpdesk ticket and return category with confidence scores."""
    if mode == "fine_tuned":
        results = classifier(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]
    else:
        # Zero-shot: use human-readable category names as candidate labels
        candidate_labels = [
            "network connectivity and VPN issues",
            "software installation and application problems",
            "hardware malfunction and device failure",
            "security incident and phishing attack",
            "account access and password reset",
            "email and calendar issues",
            "printer and printing problems",
        ]
        result = classifier(text[:512], candidate_labels=candidate_labels)
        label_to_key = {
            "network connectivity and VPN issues": "network_connectivity",
            "software installation and application problems": "software_installation",
            "hardware malfunction and device failure": "hardware_malfunction",
            "security incident and phishing attack": "security_incident",
            "account access and password reset": "account_access",
            "email and calendar issues": "email_issues",
            "printer and printing problems": "printer_problems",
        }
        scores = {
            label_to_key[label]: score
            for label, score in zip(result["labels"], result["scores"])
        }
        top_label = label_to_key[result["labels"][0]]
        top_score = result["scores"][0]

    return {
        "category": top_label,
        "display_name": CATEGORY_DISPLAY.get(top_label, top_label),
        "confidence": round(top_score * 100, 1),
        "description": CATEGORY_INFO.get(top_label, ""),
        "all_scores": scores,
        "mode": mode,
    }
