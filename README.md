---
title: IT Service Intelligence Hub
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: mit
---

# IT Service Intelligence Hub

**BITS Pilani AIMLZG 549 — Assignment II**

An API-driven AI platform for IT Service Management using Natural Language Processing.

## Sub-tasks (NLP Domain)
1. **Ticket Classification** — DistilBERT (zero-shot / fine-tuned)
2. **Sentiment Analysis** — DistilBERT-SST2
3. **Text Summarization** — DistilBART-CNN

## Extra Components
- **RAG Chatbot** — FAISS + sentence-transformers + DistilBERT-SQuAD
- **Fine-Tuning** — DistilBERT on IT helpdesk ticket dataset

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fine-Tuning (optional, improves classifier)

```bash
python fine_tuning/fine_tune.py
```

## Deployment — Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select **Streamlit** as the SDK
3. Upload all project files
4. The app auto-starts (models download on first run, ~2-3 min)
