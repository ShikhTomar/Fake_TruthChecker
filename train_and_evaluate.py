"""Model Training, Hyperparameter Tuning, and Evaluation for XLM-RoBERTa NLI on FactDrill.

This script executes the complete machine learning workflow:
1. Dataset preparation and stratified train/val/test split from FactDrill evidence corpus.
2. Cross-lingual representation extraction using pretrained XLM-RoBERTa (symanto/xlm-roberta-base-snli-mnli-anli-xnli).
3. Baseline neural classification model training.
4. Systematic hyperparameter tuning (Grid Search over learning rates, weight decays, and dropout rates).
5. Rigorous model evaluation (Accuracy, Precision, Recall, Macro F1, Weighted F1, Confusion Matrix).
6. Generation and saving of academic-grade evaluation plots in results/plots/.
7. Persistence of model checkpoints and metrics in results/.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

# Utilize available CPU cores for fast linear algebra
num_threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(num_threads)

# Set seeds for deterministic reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "factdrill_evidence_corpus.csv"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}


# ============================================================
# 1. DATASET PREPARATION FROM FACTDRILL CORPUS
# ============================================================

def build_factdrill_nli_dataset(corpus_path: Path, max_samples_per_class: int = 80) -> pd.DataFrame:
    """Extract and construct balanced NLI pairs directly from the FactDrill evidence corpus."""
    print(f"Loading FactDrill evidence corpus from {corpus_path}...", flush=True)
    df = pd.read_csv(corpus_path, low_memory=False)
    df = df.dropna(subset=["title", "investigation"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["investigation"] = df["investigation"].astype(str).str.strip()
    df = df[(df["title"].str.len() > 10) & (df["investigation"].str.len() > 50)]

    contradiction_patterns = re.compile(
        r"\b(false|fake|debunk|morphed|hoax|untrue|incorrect|misleading|not true|no evidence|photoshop|fabricated)\b",
        re.IGNORECASE,
    )
    entailment_patterns = re.compile(
        r"\b(true|confirmed|verified|accurate|real|official|actual|fact|correct)\b",
        re.IGNORECASE,
    )

    contradictions = []
    entailments = []

    for _, row in df.iterrows():
        title = row["title"]
        investigation = row["investigation"][:500]

        if contradiction_patterns.search(title) or contradiction_patterns.search(investigation[:250]):
            contradictions.append({
                "premise": investigation,
                "hypothesis": title,
                "label": "CONTRADICTION",
                "label_id": LABEL2ID["CONTRADICTION"],
                "topic": str(row.get("topic", "General")),
            })
        elif entailment_patterns.search(title) or entailment_patterns.search(investigation[:250]):
            entailments.append({
                "premise": investigation,
                "hypothesis": title,
                "label": "ENTAILMENT",
                "label_id": LABEL2ID["ENTAILMENT"],
                "topic": str(row.get("topic", "General")),
            })

    n_contra = min(len(contradictions), max_samples_per_class)
    n_ent = min(len(entailments), max_samples_per_class)
    selected_contra = random.sample(contradictions, n_contra)
    selected_ent = random.sample(entailments, n_ent)

    n_neutral = min(n_contra, n_ent, max_samples_per_class)
    neutrals = []
    shuffled_rows = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    for i in range(len(shuffled_rows) - 1):
        if len(neutrals) >= n_neutral:
            break
        row_a = shuffled_rows.iloc[i]
        row_b = shuffled_rows.iloc[i + 1]
        if str(row_a.get("topic")) != str(row_b.get("topic")):
            neutrals.append({
                "premise": str(row_a["investigation"])[:500],
                "hypothesis": str(row_b["title"]),
                "label": "NEUTRAL",
                "label_id": LABEL2ID["NEUTRAL"],
                "topic": "Cross-Topic Neutral",
            })

    all_pairs = selected_ent + neutrals + selected_contra
    random.shuffle(all_pairs)
    dataset_df = pd.DataFrame(all_pairs)
    print(f"Dataset assembled: Total samples = {len(dataset_df)}", flush=True)
    print("Class distribution:\n", dataset_df["label"].value_counts(), flush=True)
    return dataset_df


def extract_embeddings(df: pd.DataFrame, tokenizer: AutoTokenizer, encoder: AutoModel, batch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dense 768-dim pooled embeddings using the pretrained XLM-RoBERTa encoder."""
    premises = df["premise"].tolist()
    hypotheses = df["hypothesis"].tolist()
    labels = torch.tensor(df["label_id"].tolist(), dtype=torch.long)
    all_embeddings = []

    encoder.eval()
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Extracting XLM-RoBERTa representations for {len(df)} samples ({total_batches} batches)...", flush=True)
    
    with torch.no_grad():
        for b_idx, i in enumerate(range(0, len(df), batch_size), start=1):
            b_prem = premises[i : i + batch_size]
            b_hyp = hypotheses[i : i + batch_size]
            inputs = tokenizer(
                b_prem,
                b_hyp,
                truncation=True,
                padding=True,
                max_length=96,
                return_tensors="pt",
            ).to(DEVICE)
            outputs = encoder(**inputs)
            token_emb = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            sum_emb = torch.sum(token_emb * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_emb / sum_mask
            all_embeddings.append(pooled.cpu())
            print(f"  Batch {b_idx}/{total_batches} processed", flush=True)

    return torch.cat(all_embeddings, dim=0), labels


# ============================================================
# 2. NEURAL CLASSIFIER HEAD
# ============================================================

class NLIClassifierHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 3. TRAINING & EVALUATION HELPERS
# ============================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            if criterion:
                total_loss += criterion(logits, y).item() * len(y)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset) if criterion else 0.0
    return avg_loss, np.array(all_preds), np.array(all_targets)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ============================================================
# 4. MAIN PIPELINE EXECUTION
# ============================================================

def main() -> None:
    print("=" * 80, flush=True)
    print("TRUTHLENS: XLM-R NLI MODEL TRAINING & HYPERPARAMETER TUNING", flush=True)
    print("=" * 80, flush=True)
    print(f"PyTorch Device: {DEVICE} (Threads: {torch.get_num_threads()})", flush=True)

    # Step A: Dataset Preparation
    df = build_factdrill_nli_dataset(CORPUS_PATH, max_samples_per_class=80)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label_id"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=SEED, stratify=train_val_df["label_id"])

    print(f"Dataset Split: Train = {len(train_df)}, Validation = {len(val_df)}, Test = {len(test_df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    X_train, y_train = extract_embeddings(train_df, tokenizer, encoder, batch_size=16)
    X_val, y_val = extract_embeddings(val_df, tokenizer, encoder, batch_size=16)
    X_test, y_test = extract_embeddings(test_df, tokenizer, encoder, batch_size=16)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Step B: Baseline Model Training
    print("\n--- Training Baseline Model ---", flush=True)
    baseline_lr = 1e-3
    baseline_epochs = 15
    baseline_model = NLIClassifierHead(input_dim=768, hidden_dim=256, num_classes=3, dropout_rate=0.2).to(DEVICE)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=baseline_lr, weight_decay=0.01)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, baseline_epochs + 1):
        tr_loss = train_epoch(baseline_model, train_loader, baseline_optimizer, criterion)
        v_loss, v_preds, v_targets = evaluate(baseline_model, val_loader, criterion)
        v_acc = accuracy_score(v_targets, v_preds)
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        val_accuracies.append(v_acc)
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{baseline_epochs} - Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}", flush=True)

    _, base_test_preds, base_test_targets = evaluate(baseline_model, test_loader)
    baseline_metrics = compute_metrics(base_test_targets, base_test_preds)
    print("\nBaseline Model Test Results:", flush=True)
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)

    # Step C: Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning (Systematic Grid Search) ---", flush=True)
    param_grid = [
        {"name": "Config-1 (lr=3e-4, wd=0.001, drop=0.1, hdim=128)", "lr": 3e-4, "weight_decay": 0.001, "dropout": 0.1, "hidden_dim": 128, "epochs": 20},
        {"name": "Config-2 (lr=5e-4, wd=0.01, drop=0.2, hdim=256)", "lr": 5e-4, "weight_decay": 0.01, "dropout": 0.2, "hidden_dim": 256, "epochs": 20},
        {"name": "Config-3 (lr=1e-3, wd=0.01, drop=0.2, hdim=256)", "lr": 1e-3, "weight_decay": 0.01, "dropout": 0.2, "hidden_dim": 256, "epochs": 20},
        {"name": "Config-4 (lr=2e-3, wd=0.05, drop=0.3, hdim=512)", "lr": 2e-3, "weight_decay": 0.05, "dropout": 0.3, "hidden_dim": 512, "epochs": 20},
    ]

    tuning_results = []
    best_val_f1 = -1.0
    best_config = None
    best_model = None

    for config in param_grid:
        model = NLIClassifierHead(input_dim=768, hidden_dim=config["hidden_dim"], num_classes=3, dropout_rate=config["dropout"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        for _ in range(config["epochs"]):
            train_epoch(model, train_loader, optimizer, criterion)

        v_loss, v_preds, v_targets = evaluate(model, val_loader, criterion)
        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average="macro")

        print(f"  {config['name']} --> Val Acc: {v_acc:.4f} | Val Macro F1: {v_f1:.4f}", flush=True)
        tuning_results.append({
            "config": config["name"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "dropout": config["dropout"],
            "hidden_dim": config["hidden_dim"],
            "val_loss": float(v_loss),
            "val_accuracy": float(v_acc),
            "val_f1_macro": float(v_f1),
        })

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_config = config
            best_model = model

    print(f"\nBest Hyperparameter Set Selected: {best_config['name']} (Val Macro F1 = {best_val_f1:.4f})", flush=True)

    # Step D: Model Evaluation on Test Set
    print("\n--- Final Model Evaluation on Held-Out Test Set ---", flush=True)
    _, tuned_test_preds, tuned_test_targets = evaluate(best_model, test_loader)
    tuned_metrics = compute_metrics(tuned_test_targets, tuned_test_preds)

    print("Tuned Model Test Set Results vs Baseline:", flush=True)
    for k, v in tuned_metrics.items():
        delta = v - baseline_metrics[k]
        print(f"  {k}: {v:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})", flush=True)

    print("\nDetailed Classification Report (Tuned XLM-R Head):", flush=True)
    print(classification_report(tuned_test_targets, tuned_test_preds, target_names=LABELS, digits=4), flush=True)

    # Save Checkpoint
    model_save_path = MODELS_DIR / "best_xlm_r_head.pt"
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "best_config": best_config,
        "labels": LABELS,
        "metrics": tuned_metrics,
    }, model_save_path)
    print(f"Checkpoint saved to: {model_save_path}", flush=True)

    # Step E: Graphical Result Analysis
    print("\n--- Generating Professional Figures ---", flush=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Training & Validation Curves
    plt.figure(figsize=(9, 4.5))
    epochs_range = list(range(1, len(train_losses) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, "o-", label="Training Loss", color="#4f46e5", linewidth=2)
    plt.plot(epochs_range, val_losses, "s--", label="Validation Loss", color="#e11d48", linewidth=2)
    plt.title("Cross-Entropy Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in val_accuracies], "d-", label="Validation Accuracy", color="#059669", linewidth=2)
    plt.title("Validation Accuracy Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    curve_plot_path = PLOTS_DIR / "training_validation_curves.png"
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {curve_plot_path}", flush=True)

    # 2. Hyperparameter Tuning Comparison
    plt.figure(figsize=(10, 5))
    cfgs = [f"Cfg-{i+1}" for i in range(len(tuning_results))]
    val_accs = [r["val_accuracy"] * 100 for r in tuning_results]
    val_f1s = [r["val_f1_macro"] * 100 for r in tuning_results]
    x = np.arange(len(cfgs))
    width = 0.35

    plt.bar(x - width/2, val_accs, width, label="Val Accuracy (%)", color="#6366f1")
    plt.bar(x + width/2, val_f1s, width, label="Val Macro F1 (%)", color="#10b981")
    plt.xticks(x, [f"{r['config'][:22]}..." for r in tuning_results], rotation=15, ha="right", fontsize=9)
    plt.ylabel("Score (%)")
    plt.title("Hyperparameter Search: Validation Performance Comparison", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    tuning_plot_path = PLOTS_DIR / "hyperparameter_tuning_comparison.png"
    plt.savefig(tuning_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {tuning_plot_path}", flush=True)

    # 3. Baseline vs. Tuned Metrics Comparison
    plt.figure(figsize=(9, 5))
    metric_names = ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]
    base_scores = [
        baseline_metrics["accuracy"] * 100,
        baseline_metrics["precision_macro"] * 100,
        baseline_metrics["recall_macro"] * 100,
        baseline_metrics["f1_macro"] * 100,
        baseline_metrics["f1_weighted"] * 100,
    ]
    tuned_scores = [
        tuned_metrics["accuracy"] * 100,
        tuned_metrics["precision_macro"] * 100,
        tuned_metrics["recall_macro"] * 100,
        tuned_metrics["f1_macro"] * 100,
        tuned_metrics["f1_weighted"] * 100,
    ]
    x = np.arange(len(metric_names))
    width = 0.35

    plt.bar(x - width/2, base_scores, width, label="Baseline Model", color="#94a3b8")
    plt.bar(x + width/2, tuned_scores, width, label="Tuned XLM-R Head", color="#4338ca")
    for i in range(len(metric_names)):
        plt.text(x[i] - width/2, base_scores[i] + 1, f"{base_scores[i]:.1f}%", ha="center", fontsize=8)
        plt.text(x[i] + width/2, tuned_scores[i] + 1, f"{tuned_scores[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.xticks(x, metric_names, fontweight="bold")
    plt.ylabel("Performance Score (%)")
    plt.ylim(0, 105)
    plt.title("FactDrill Test Set: Baseline vs. Tuned XLM-R NLI", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_plot_path = PLOTS_DIR / "baseline_vs_tuned_metrics.png"
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_plot_path}", flush=True)

    # 4. Confusion Matrices (Baseline & Tuned)
    cm_base = confusion_matrix(base_test_targets, base_test_preds)
    cm_tuned = confusion_matrix(tuned_test_targets, tuned_test_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Baseline Model Confusion Matrix", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Ground-Truth")
    plt.tight_layout()
    cm_base_path = PLOTS_DIR / "confusion_matrix_baseline.png"
    plt.savefig(cm_base_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_base_path}", flush=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Tuned XLM-R NLI Confusion Matrix", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Ground-Truth")
    plt.tight_layout()
    cm_tuned_path = PLOTS_DIR / "confusion_matrix_tuned.png"
    plt.savefig(cm_tuned_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_tuned_path}", flush=True)

    # Save summary metrics to json
    all_summary = {
        "model_name": MODEL_NAME,
        "dataset": "FactDrill Evidence Corpus",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "baseline_metrics": baseline_metrics,
        "best_config": best_config,
        "tuning_results": tuning_results,
        "tuned_metrics": tuned_metrics,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    print(f"Summary metrics saved to: {metrics_path}", flush=True)
    print("\nTraining and evaluation workflow successfully executed!", flush=True)


if __name__ == "__main__":
    main()
