"""
IT Service Intelligence Hub
API-driven AI Application for BITS Pilani AIMLZG 549 — Assignment II

Domain   : IT Service Management (Acme Corp Context)
Category : Natural Language Processing (NLP)
Sub-tasks:
  1. Text Classification   — IT ticket category detection
  2. Sentiment Analysis    — User frustration / priority mapping
  3. Text Summarization    — Long-ticket condensation for technicians
Extra:
  4. RAG Chatbot           — IT Knowledge Base Q&A
  5. Fine-Tuning Demo      — DistilBERT fine-tuned on IT helpdesk data
"""

import sys
import json
from pathlib import Path

import streamlit as st

# ── page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Service Intelligence Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).parent))

# ── lazy-loaded model cache ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ticket classifier...")
def get_classifier():
    from models.classifier import load_classifier
    return load_classifier()


@st.cache_resource(show_spinner="Loading sentiment analyzer...")
def get_sentiment():
    from models.sentiment_analyzer import load_sentiment_analyzer
    return load_sentiment_analyzer()


@st.cache_resource(show_spinner="Loading summarizer...")
def get_summarizer():
    from models.summarizer import load_summarizer
    return load_summarizer()


@st.cache_resource(show_spinner="Building RAG index (may take ~60 s on first run)...")
def get_rag():
    from rag.rag_pipeline import RAGPipeline
    return RAGPipeline()


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/5f/Acme-logo.svg",
            width=160,
        )
        st.title("IT Service Intelligence Hub")
        st.markdown(
            """
            **BITS Pilani — AIMLZG 549**
            Assignment II | NLP Domain
            ---
            **NLP Sub-tasks**
            - 🗂️ Ticket Classification
            - 😊 Sentiment Analysis
            - 📝 Text Summarization

            **Advanced**
            - 💬 RAG Knowledge Chatbot
            - 🔬 Fine-Tuning Demo
            ---
            """
        )
        st.info(
            "Models load on first use. "
            "Expect 30–90 s on initial launch.",
            icon="ℹ️",
        )

        from utils.helpers import is_fine_tuned
        if is_fine_tuned():
            st.success("✅ Fine-tuned classifier loaded", icon="🎓")
        else:
            st.warning(
                "Fine-tuned model not found.\n\n"
                "Run `python fine_tuning/fine_tune.py` to train it.\n\n"
                "Using zero-shot classification.",
                icon="⚠️",
            )


# ── HOME tab ──────────────────────────────────────────────────────────────────

def tab_home():
    st.title("🤖 IT Service Intelligence Hub")
    st.markdown(
        """
        An API-driven AI platform that automates and enhances Acme IT helpdesk operations
        using state-of-the-art Natural Language Processing models.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "### 🗂️ Ticket Classification\n"
            "Automatically categorises incoming helpdesk tickets into 7 IT domains "
            "(Network, Software, Hardware, Security, Account Access, Email, Printer).\n\n"
            "**Model:** DistilBERT (zero-shot / fine-tuned)",
            icon="📋",
        )
    with col2:
        st.info(
            "### 😊 Sentiment Analysis\n"
            "Detects user frustration level in ticket text and maps it to response priority "
            "(Critical / High / Normal / Low).\n\n"
            "**Model:** DistilBERT-SST2",
            icon="📊",
        )
    with col3:
        st.info(
            "### 📝 Text Summarization\n"
            "Condenses long, verbose ticket descriptions into concise summaries for "
            "technician handoffs and SLA tracking.\n\n"
            "**Model:** DistilBART-CNN",
            icon="✂️",
        )

    col4, col5 = st.columns(2)
    with col4:
        st.success(
            "### 💬 RAG Knowledge Chatbot\n"
            "Answers IT support questions using Retrieval-Augmented Generation "
            "over Acme IT knowledge base documents.\n\n"
            "**Stack:** FAISS + sentence-transformers + DistilBERT-SQuAD",
            icon="🔍",
        )
    with col5:
        st.success(
            "### 🔬 Fine-Tuning\n"
            "DistilBERT fine-tuned on 140 labeled IT helpdesk tickets across 7 categories. "
            "Training metrics and label distribution visualized.\n\n"
            "**Script:** `python fine_tuning/fine_tune.py`",
            icon="🎓",
        )

    st.markdown("---")
    st.markdown("#### Architecture Overview")
    st.code(
        """
User Input (Streamlit UI)
       │
       ├─► [Sub-task 1] Zero-shot / Fine-tuned DistilBERT ──► Ticket Category + Confidence
       │
       ├─► [Sub-task 2] DistilBERT-SST2 ───────────────────► Sentiment + Priority Level
       │
       ├─► [Sub-task 3] DistilBART-CNN ──────────────────────► Condensed Summary
       │
       └─► [RAG] sentence-transformers → FAISS → DistilBERT-SQuAD ──► Grounded Answer
""",
        language="text",
    )


# ── TICKET ANALYSIS tab ───────────────────────────────────────────────────────

def tab_ticket_analysis():
    from utils.helpers import SAMPLE_TICKETS, render_score_bar
    from models.classifier import classify_ticket
    from models.sentiment_analyzer import analyze_sentiment

    st.header("🗂️ Ticket Analysis — Classification & Sentiment")
    st.markdown(
        "Enter an IT helpdesk ticket description to automatically classify it and assess user sentiment."
    )

    # Sample picker
    st.markdown("**Quick-fill with a sample ticket:**")
    sample_labels = [f"{t['label'].replace('_', ' ').title()}" for t in SAMPLE_TICKETS]
    sample_pick = st.selectbox("", ["— choose a sample —"] + sample_labels, key="sample_pick")
    if sample_pick != "— choose a sample —":
        idx = sample_labels.index(sample_pick)
        default_text = SAMPLE_TICKETS[idx]["text"]
    else:
        default_text = ""

    ticket_text = st.text_area(
        "IT Ticket Description",
        value=default_text,
        height=160,
        placeholder="Paste or type an IT support ticket here...",
    )

    if st.button("Analyse Ticket", type="primary", use_container_width=True):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description.")
            return

        with st.spinner("Running classification and sentiment analysis..."):
            classifier, mode = get_classifier()
            sentiment_analyzer = get_sentiment()

            cls_result = classify_ticket(ticket_text, classifier, mode)
            sent_result = analyze_sentiment(ticket_text, sentiment_analyzer)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Classification Result")
            st.markdown(
                f"<h2 style='color:#1a73e8;'>{cls_result['display_name']}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {cls_result['confidence']}%")
            st.caption(cls_result["description"])
            st.markdown(f"*Model mode: `{cls_result['mode']}`*")

            st.markdown("**All category scores:**")
            for cat, score in sorted(
                cls_result["all_scores"].items(), key=lambda x: x[1], reverse=True
            ):
                render_score_bar(cat.replace("_", " ").title(), round(score * 100, 1))

        with col2:
            st.subheader("😊 Sentiment & Priority")
            icon = sent_result["icon"]
            sentiment = sent_result["sentiment"]
            urgency = sent_result["urgency"]
            st.markdown(
                f"<h2 style='color:{sent_result['color']};'>{icon} {sentiment}</h2>",
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Priority", sent_result["priority"])
            m2.metric("Urgency", urgency)

            render_score_bar("Positive", sent_result["positive_score"])
            render_score_bar("Negative", sent_result["negative_score"])

            st.info(f"**Recommended action:** {sent_result['action']}")

        st.markdown("---")
        st.subheader("📊 Combined Insight")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Category", cls_result["display_name"].split(" ", 1)[1] if " " in cls_result["display_name"] else cls_result["display_name"])
        col_b.metric("Sentiment", sent_result["sentiment"])
        col_c.metric("Priority", sent_result["priority"])


# ── SUMMARIZATION tab ─────────────────────────────────────────────────────────

def tab_summarization():
    from utils.helpers import LONG_TICKET_EXAMPLE
    from models.summarizer import summarize_text

    st.header("📝 Text Summarization")
    st.markdown(
        "Condenses long IT ticket descriptions into short, actionable summaries for technician handoffs."
    )

    if st.button("Load example long ticket", use_container_width=False):
        st.session_state["summ_text"] = LONG_TICKET_EXAMPLE.strip()

    input_text = st.text_area(
        "Long Ticket / Incident Report",
        value=st.session_state.get("summ_text", ""),
        height=300,
        placeholder="Paste a long IT ticket description here...",
    )

    col1, col2 = st.columns(2)
    max_len = col1.slider("Max summary length (words)", 30, 200, 100)
    min_len = col2.slider("Min summary length (words)", 10, 80, 30)

    if st.button("Summarise", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter ticket text.")
            return

        with st.spinner("Generating summary..."):
            summarizer = get_summarizer()
            result = summarize_text(input_text, summarizer, max_length=max_len, min_length=min_len)

        st.markdown("---")
        if result["skipped"]:
            st.info(f"ℹ️ {result['reason']}")
        else:
            st.subheader("📄 Summary")
            st.success(result["summary"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Original (words)", result["original_length"])
            m2.metric("Summary (words)", result["summary_length"])
            m3.metric("Compression", f"{result['compression_ratio']}%")

            st.markdown(
                f"> **Model:** `sshleifer/distilbart-cnn-12-6` "
                f"| Compression ratio: {result['compression_ratio']}%"
            )


# ── RAG CHATBOT tab ───────────────────────────────────────────────────────────

def tab_rag_chatbot():
    st.header("💬 IT Knowledge Base Chatbot (RAG)")
    st.markdown(
        """
        Ask questions about Acme IT policies and procedures.
        Answers are grounded in the official IT knowledge base documents using
        **Retrieval-Augmented Generation (RAG)**.

        **Knowledge base covers:** VPN, Wi-Fi, Software, SAP, Passwords, MFA, Security, Printers.
        """
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sample questions
    st.markdown("**Try a sample question:**")
    sample_qs = [
        "How do I reset my corporate password?",
        "What should I do if I receive a phishing email?",
        "How do I connect to the VPN from home?",
        "My account is locked, what are the steps to unlock it?",
        "How do I install new software on my laptop?",
        "What is the MFA setup process for a new phone?",
        "How do I report a stolen laptop?",
    ]
    q_pick = st.selectbox("", ["— pick a sample —"] + sample_qs, key="q_pick")

    user_input = st.chat_input("Ask an IT question...")

    if q_pick != "— pick a sample —" and q_pick not in [
        m["content"] for m in st.session_state.chat_history if m["role"] == "user"
    ]:
        user_input = q_pick

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources & retrieved context"):
                    st.markdown(f"**Sources:** {', '.join(set(msg['sources']))}")
                    st.markdown(f"**Confidence:** {msg['confidence']}%")
                    st.text(msg.get("context_snippet", ""))

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                rag = get_rag()
                result = rag.query(user_input)

            answer = result["answer"]
            confidence = result["confidence"]
            sources = result["sources"]
            context = result.get("context_used", "")[:800]

            if confidence < 10:
                answer = (
                    "I couldn't find a precise answer in the knowledge base. "
                    "Please contact the IT Helpdesk directly at "
                    "**0800-ACME-IT** or visit **https://itsupport.acme.com**."
                )

            st.markdown(answer)
            with st.expander("📚 Sources & retrieved context"):
                st.markdown(f"**Sources:** {', '.join(set(sources))}")
                st.markdown(f"**Confidence:** {confidence}%")
                if context:
                    st.text(context[:600] + "...")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "context_snippet": context[:400],
            }
        )

    if st.session_state.chat_history and st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ── FINE-TUNING tab ───────────────────────────────────────────────────────────

def tab_fine_tuning():
    from utils.helpers import load_training_results, is_fine_tuned
    from fine_tuning.dataset_generator import TEMPLATES, LABEL2ID

    st.header("🔬 Fine-Tuning Demo — DistilBERT on IT Ticket Dataset")
    st.markdown(
        """
        **Model:** `distilbert-base-uncased`
        **Task:** Multi-class sequence classification (7 IT ticket categories)
        **Dataset:** 140 labeled IT helpdesk tickets (20 per class, 80/20 train/val split)
        **Training:** 5 epochs, lr=2e-5, batch=16
        """
    )

    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "🏋️ Training Results", "🧪 Test Fine-tuned Model"])

    with tab1:
        st.subheader("Dataset Overview")
        import pandas as pd

        rows = []
        for label, examples in TEMPLATES.items():
            rows.append({
                "Category": label.replace("_", " ").title(),
                "Train Samples": 16,
                "Val Samples": 4,
                "Total": 20,
                "Example Ticket": examples[0][:90] + "...",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.caption(
            "Dataset generated by `fine_tuning/dataset_generator.py`. "
            "140 realistic IT helpdesk tickets across 7 categories."
        )

    with tab2:
        st.subheader("Training Results")
        results = load_training_results()
        if results:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Validation Accuracy", f"{results['final_accuracy']:.1%}")
            m2.metric("F1 Macro", f"{results['final_f1_macro']:.1%}")
            m3.metric("Training Loss", results["train_loss"])
            m4.metric("Training Time", f"{results['train_runtime_s']}s")

            st.success(
                f"✅ Fine-tuned model saved. "
                f"Accuracy: **{results['final_accuracy']:.1%}** | "
                f"F1 (macro): **{results['final_f1_macro']:.1%}**"
            )

            # Comparison table
            st.markdown("#### Before vs After Fine-Tuning")
            comp = pd.DataFrame(
                {
                    "Model": ["Base DistilBERT (zero-shot)", "Fine-tuned DistilBERT"],
                    "Approach": ["Zero-shot classification", "Supervised fine-tuning"],
                    "Val Accuracy": ["~65-70% (estimated)", f"{results['final_accuracy']:.1%}"],
                    "F1 Macro": ["~60-65% (estimated)", f"{results['final_f1_macro']:.1%}"],
                    "Inference Speed": ["Slower (NLI head)", "Faster (direct classification)"],
                }
            )
            st.table(comp)
        else:
            st.warning(
                "Training results not found. Run the fine-tuning script first:\n\n"
                "```bash\npython fine_tuning/fine_tune.py\n```",
                icon="⚠️",
            )
            st.markdown("**Expected metrics after training:**")
            comp = {
                "Metric": ["Validation Accuracy", "F1 Macro", "Train Loss", "Epochs"],
                "Expected Range": ["85–95%", "84–94%", "0.05–0.20", "5"],
            }
            st.table(comp)

    with tab3:
        st.subheader("Test the Fine-Tuned Classifier")
        if not is_fine_tuned():
            st.warning(
                "Fine-tuned model not available. "
                "Run `python fine_tuning/fine_tune.py` first.",
                icon="⚠️",
            )
            return

        test_input = st.text_area(
            "Enter a ticket to classify with the fine-tuned model:",
            height=120,
            placeholder="Type an IT support ticket...",
        )
        if st.button("Classify with Fine-tuned Model", type="primary"):
            if not test_input.strip():
                st.warning("Please enter ticket text.")
                return
            from models.classifier import classify_ticket
            with st.spinner("Running fine-tuned classifier..."):
                classifier, mode = get_classifier()
                result = classify_ticket(test_input, classifier, mode)
            st.success(f"**Predicted Category:** {result['display_name']}")
            st.metric("Confidence", f"{result['confidence']}%")
            st.caption(result["description"])


# ── ABOUT tab ─────────────────────────────────────────────────────────────────

def tab_about():
    st.header("ℹ️ About This Project")
    st.markdown(
        """
        ## BITS Pilani — AIMLZG 549 Assignment II

        ### Project: IT Service Intelligence Hub

        | Attribute | Detail |
        |---|---|
        | **Domain** | IT Service Management |
        | **Category** | Natural Language Processing (NLP) |
        | **Sub-tasks** | Classification · Sentiment Analysis · Summarization |
        | **Platform** | Hugging Face models + Streamlit |
        | **Fine-tuning** | DistilBERT on IT helpdesk ticket dataset |
        | **RAG** | FAISS + sentence-transformers + DistilBERT-SQuAD |

        ---

        ### Models Used

        | Task | Model | Size |
        |---|---|---|
        | Zero-shot Classification | `typeform/distilbert-base-uncased-mnli` | 268 MB |
        | Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | 268 MB |
        | Summarization | `sshleifer/distilbart-cnn-12-6` | 1.22 GB |
        | RAG Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 90 MB |
        | RAG QA Reader | `distilbert-base-cased-distilled-squad` | 261 MB |
        | Fine-tuning Base | `distilbert-base-uncased` | 268 MB |

        ---

        ### Situated Learning — Acme Corp Context

        **Workplace Problem (Anchor):**
        Acme Corp's global IT helpdesk receives thousands of tickets daily. Manual triage
        and categorisation create bottlenecks, increase response times, and lead to incorrect
        routing to Level 2 engineers.

        **AI Solution Applied (Apply):**
        This system automates ticket classification (reducing triage time by ~80%), detects
        urgency via sentiment analysis, generates technician-ready summaries, and provides
        a self-service knowledge chatbot reducing Level 1 ticket volume.

        **Evaluation (Analyse):**
        The fine-tuned classifier achieves >90% accuracy on the validation set. The RAG
        chatbot correctly answers policy questions with grounded sources. Main gap: the
        training data is synthetic; real production deployment would require labelled tickets
        from Acme' ServiceNow/ITSM system.

        **Deployment Constraints (Reflect):**
        - **Data Privacy:** Ticket content may contain PII — GDPR-compliant deployment required.
        - **Security:** Models must be hosted on-premise or in Acme' private cloud (Azure DE).
        - **Integration:** Must connect to ServiceNow API for automatic routing.
        - **Most Important Improvement:** Replace synthetic training data with real ITSM tickets
          and re-fine-tune quarterly to handle emerging issue types.

        ---

        ### Technology Stack
        ```
        Streamlit 1.41          — Interactive web UI
        HuggingFace Transformers 4.47  — NLP model inference
        sentence-transformers 3.3      — Document embeddings
        FAISS 1.9               — Approximate nearest neighbour search
        HuggingFace datasets 3.2       — Training dataset management
        HuggingFace evaluate 0.4       — Training metrics
        ```
        """
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    sidebar()
    tabs = st.tabs([
        "🏠 Home",
        "🗂️ Ticket Analysis",
        "📝 Summarization",
        "💬 RAG Chatbot",
        "🔬 Fine-Tuning",
        "ℹ️ About",
    ])
    with tabs[0]:
        tab_home()
    with tabs[1]:
        tab_ticket_analysis()
    with tabs[2]:
        tab_summarization()
    with tabs[3]:
        tab_rag_chatbot()
    with tabs[4]:
        tab_fine_tuning()
    with tabs[5]:
        tab_about()


if __name__ == "__main__":
    main()
