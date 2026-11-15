"""
Fine-Tuning Script — IT Ticket Classifier
Fine-tunes distilbert-base-uncased on the IT helpdesk ticket dataset.

Usage:
    python fine_tuning/fine_tune.py

The fine-tuned model is saved to: fine_tuning/saved_model/
Run from the project root directory.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from fine_tuning.dataset_generator import build_dataset, LABEL2ID, ID2LABEL

BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = Path(__file__).parent / "saved_model"
RESULTS_FILE = Path(__file__).parent / "training_results.json"

NUM_LABELS = len(LABEL2ID)


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding=False,
        )
    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1_score}


def train():
    print("=" * 60)
    print("IT Ticket Classifier — Fine-Tuning")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    # Load dataset
    print("\n[1/4] Generating dataset...")
    dataset = build_dataset(samples_per_class=20)
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Val:   {len(dataset['validation'])} samples")

    # Tokenize
    print("\n[2/4] Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenized = tokenize_dataset(dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    print("\n[3/4] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n[4/4] Training...")
    train_result = trainer.train()
    metrics = trainer.evaluate()

    # Save
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save results for the app to display
    results = {
        "base_model": BASE_MODEL,
        "num_labels": NUM_LABELS,
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["validation"]),
        "epochs": args.num_train_epochs,
        "final_accuracy": round(metrics.get("eval_accuracy", 0), 4),
        "final_f1_macro": round(metrics.get("eval_f1_macro", 0), 4),
        "train_loss": round(train_result.training_loss, 4),
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 1),
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Validation Accuracy : {results['final_accuracy']:.2%}")
    print(f"  Validation F1 Macro : {results['final_f1_macro']:.2%}")
    print(f"  Model saved to      : {OUTPUT_DIR}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train()
