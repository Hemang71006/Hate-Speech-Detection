"""
transformer_pipeline.py
========================
Fine-tune a HuggingFace transformer (BERT / RoBERTa) for 3-class
hate-speech detection on the Davidson dataset.

Labels
------
  0 = Hate Speech
  1 = Offensive Language
  2 = Neutral / Neither

Usage (from the project folder with venv active)
-------------------------------------------------
  python transformer_pipeline.py \
      --data dataset/data/labeled_data.csv \
      --model bert-base-uncased \
      --epochs 3 \
      --batch-size 16 \
      --lr 2e-5 \
      --run-name transformer_bert

All artefacts (model weights, tokenizer, metrics, confusion-matrix
plots, comparison table) are saved to  artifacts/<run-name>/ .
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend so the script works on headless servers / Colab
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────
# 1.  HELPERS
# ────────────────────────────────────────────────────────────

LABEL_MAP = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}


def _resolve_data_path(path: str | None) -> str:
    """Return a valid CSV path or raise an error."""
    if path and os.path.exists(path):
        return path
    candidates = [
        "dataset/data/labeled_data.csv",
        "dataset/labeled_data.csv",
        "data/labeled_data.csv",
        "labeled_data.csv",
    ]
    found = next((p for p in candidates if os.path.exists(p)), None)
    if not found:
        raise FileNotFoundError(
            "Could not find labeled_data.csv — provide the path with --data"
        )
    return found


def light_clean(text: str) -> str:
    """
    Minimal text cleaning for transformer input.

    We keep more text than the TF-IDF pipeline because transformers
    understand context; we only strip URLs, @-mentions, and extra whitespace.
    """
    text = re.sub(r"https?://\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                # remove @mentions
    text = re.sub(r"#", "", text)                   # strip # but keep hashtag word
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def make_run_dir(base: str = "artifacts", run_name: str | None = None) -> Path:
    """Create a unique run directory and return the Path object."""
    tag = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ────────────────────────────────────────────────────────────
# 2.  LOAD & PREPROCESS DATASET
# ────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str, test_size: float = 0.20, seed: int = 42):
    """
    Load Davidson CSV → clean text → 80/20 stratified split → HuggingFace
    Dataset objects ready for tokenisation.

    Returns
    -------
    train_ds, val_ds : datasets.Dataset
    label_names      : list[str]
    """
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # ── find text and label columns (same logic as baseline pipeline) ──
    text_col = next(
        (c for c in df.columns if c.lower() in ("tweet", "text", "message")),
        df.select_dtypes("object").columns[0],
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("class", "label", "category")),
        df.select_dtypes("number").columns[0],
    )

    df = df[[text_col, label_col]].dropna().copy()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    df["label"] = df["label"].astype(int)

    # light cleaning — keep more context for the transformer
    df["text"] = df["text"].apply(light_clean)

    print(f"Dataset loaded: {len(df):,} samples")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index().to_string()}")

    # ── stratified 80/20 split ──
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # ── convert to HuggingFace Dataset objects ──
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    label_names = [LABEL_MAP.get(i, str(i)) for i in sorted(df["label"].unique())]
    return train_ds, val_ds, label_names


# ────────────────────────────────────────────────────────────
# 3.  TOKENIZATION
# ────────────────────────────────────────────────────────────

def tokenize_datasets(train_ds, val_ds, model_name: str, max_length: int = 128):
    """
    Load the pretrained tokenizer and tokenize both splits.
    Returns tokenized datasets + tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tok(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_ds = train_ds.map(_tok, batched=True, desc="Tokenising train")
    val_ds   = val_ds.map(_tok,   batched=True, desc="Tokenising val")

    # keep only columns the model needs
    cols_to_keep = {"input_ids", "attention_mask", "label"}
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in cols_to_keep]
    )
    val_ds = val_ds.remove_columns(
        [c for c in val_ds.column_names if c not in cols_to_keep]
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    return train_ds, val_ds, tokenizer


# ────────────────────────────────────────────────────────────
# 4.  CLASS-WEIGHT COMPUTATION  (helps minority-class recall)
# ────────────────────────────────────────────────────────────

def compute_class_weights(train_ds) -> torch.Tensor:
    """
    Compute inverse-frequency class weights so the loss function
    penalises mistakes on the minority class (Hate Speech) more heavily.
    """
    labels = np.array(train_ds["label"])
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)   # normalise
    print(f"Class weights: {dict(enumerate(weights.round(4)))}")
    return torch.tensor(weights, dtype=torch.float32)


# ────────────────────────────────────────────────────────────
# 5.  CUSTOM TRAINER WITH WEIGHTED LOSS
# ────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that injects class weights into
    the cross-entropy loss.  This is the simplest way to handle class
    imbalance without changing the dataset itself.
    """

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ────────────────────────────────────────────────────────────
# 6.  EVALUATION METRICS  (called after every epoch)
# ────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """
    Compute accuracy, macro precision, recall, f1 from the Trainer's
    EvalPrediction. These numbers appear in the training log table.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ────────────────────────────────────────────────────────────
# 7.  DETAILED EVALUATION + SAVE REPORTS
# ────────────────────────────────────────────────────────────

def full_evaluation(trainer: Trainer, val_ds, label_names, run_dir: Path):
    """
    Run the final evaluation pass and save:
      - classification_report_transformer.txt
      - confusion_transformer.png
      - metrics_transformer.json
    """
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    # ── classification report ──
    report_str = classification_report(
        labels, preds, target_names=label_names, digits=4
    )
    print("\n" + "=" * 60)
    print("TRANSFORMER — Classification Report")
    print("=" * 60)
    print(report_str)
    (run_dir / "classification_report_transformer.txt").write_text(
        report_str, encoding="utf-8"
    )

    # ── confusion matrix ──
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Transformer")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_transformer.png", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {run_dir / 'confusion_transformer.png'}")

    # ── scalar metrics ──
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    metrics = {
        "accuracy": round(acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
    (run_dir / "metrics_transformer.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Metrics saved → {run_dir / 'metrics_transformer.json'}")
    return metrics


# ────────────────────────────────────────────────────────────
# 8.  COMPARE WITH BASELINE MODELS
# ────────────────────────────────────────────────────────────

def compare_with_baselines(
    transformer_metrics: dict,
    baseline_path: str = "artifacts/class_project_01/metrics_summary.json",
    run_dir: Path | None = None,
):
    """
    Load baseline metrics (Logistic Regression, LinearSVC, LR_oversampled)
    from the earlier pipeline run and print a side-by-side comparison table.
    Also saves comparison_table.csv to run_dir.
    """
    rows = []

    # ── load baselines if available ──
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baselines = json.load(f)
        for name, m in baselines.items():
            rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
    else:
        print(f"[info] Baseline metrics not found at {baseline_path} — skipping comparison.")

    # ── add transformer row ──
    rows.append(
        {
            "Model": "Transformer",
            **{k: round(v, 4) for k, v in transformer_metrics.items()},
        }
    )

    table = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string())
    print()

    if run_dir:
        csv_path = run_dir / "comparison_table.csv"
        table.to_csv(csv_path)
        print(f"Comparison table saved → {csv_path}")

    return table


# ────────────────────────────────────────────────────────────
# 9.  MAIN — ORCHESTRATION
# ────────────────────────────────────────────────────────────

def main(args):
    # ── device selection ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("[warn] Training on CPU will be slow.  Use a GPU / Colab for faster results.")

    # ── run directory ──
    run_dir = make_run_dir(run_name=args.run_name)
    print(f"Artifacts will be saved to: {run_dir}")

    # ── step 1-2: load, clean, split, make HuggingFace datasets ──
    train_ds, val_ds, label_names = load_and_prepare(
        csv_path=_resolve_data_path(args.data),
        test_size=args.test_size,
        seed=args.seed,
    )

    # ── step 3-4: tokenize ──
    train_ds, val_ds, tokenizer = tokenize_datasets(
        train_ds, val_ds, model_name=args.model, max_length=args.max_length
    )

    # ── compute class weights for imbalance handling ──
    class_weights = compute_class_weights(train_ds)

    # ── step 5-6: load pretrained model for 3-class classification ──
    num_labels = len(label_names)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={i: n for i, n in enumerate(label_names)},
        label2id={n: i for i, n in enumerate(label_names)},
    )

    # ── step 7: training arguments ──
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=(device == "cuda"),       # mixed-precision on GPU
        report_to="none",             # no wandb / mlflow
        seed=args.seed,
    )

    # ── create trainer ──
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── train ──
    print("\n" + "=" * 60)
    print(f"Fine-tuning {args.model}  ({args.epochs} epochs, lr={args.lr})")
    print("=" * 60)
    trainer.train()

    # ── step 8: detailed evaluation ──
    metrics = full_evaluation(trainer, val_ds, label_names, run_dir)

    # ── step 9: compare with baselines ──
    compare_with_baselines(
        transformer_metrics=metrics,
        baseline_path=args.baseline_metrics,
        run_dir=run_dir,
    )

    # ── step 10: save model & tokenizer for deployment ──
    save_path = run_dir / "saved_model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"\nModel + tokenizer saved → {save_path}")

    # ── save run config for reproducibility ──
    config = {k: str(v) for k, v in vars(args).items()}
    config["device"] = device
    config["num_labels"] = num_labels
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n✓ Transformer pipeline finished successfully.")
    print(f"  All outputs are in:  {run_dir.resolve()}")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def cli():
    ap = argparse.ArgumentParser(
        description="Fine-tune a HuggingFace transformer for hate-speech detection"
    )
    ap.add_argument("--data", type=str, default=None,
                    help="Path to labeled_data.csv")
    ap.add_argument("--model", type=str, default="bert-base-uncased",
                    help="HuggingFace model name (default: bert-base-uncased)")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs (default: 3)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Batch size (default: 16)")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate (default: 2e-5)")
    ap.add_argument("--max-length", type=int, default=128,
                    help="Max token length (default: 128)")
    ap.add_argument("--test-size", type=float, default=0.20,
                    help="Validation split fraction (default: 0.20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Name for the run folder under artifacts/")
    ap.add_argument("--baseline-metrics", type=str,
                    default="artifacts/class_project_01/metrics_summary.json",
                    help="Path to baseline metrics_summary.json for comparison")
    return ap.parse_args()


if __name__ == "__main__":
    main(cli())
