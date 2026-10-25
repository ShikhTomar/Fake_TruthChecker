import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

INPUT_FILE = BASE / "realtime_ranked_evidence.csv"
OUTPUT_FILE = BASE / "realtime_nli_results.csv"

# XLM-RoBERTa cross-lingual NLI model as specified in the reference architecture
MODEL_NAME = "symanto/xlm-roberta-base-snli-mnli-anli-xnli"

# ============================================================
# LOAD TOP 3
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

df = df.head(3).copy()

print("Live evidence pairs:", len(df))

claim = str(df["claim"].iloc[0])

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading XLM-R NLI model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
model.eval()

print("Device:", device)
print("Labels:", model.config.id2label)

# ============================================================
# NLI
# ============================================================

claims = [
    claim
] * len(df)

evidence = (
    df["evidence"]
    .fillna("")
    .astype(str)
    .tolist()
)

# Evidence = premise
# Claim = hypothesis

inputs = tokenizer(
    evidence,
    claims,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

print("\nRunning NLI...")

with torch.no_grad():

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )

# ============================================================
# RESULTS
# ============================================================

labels = []
confidences = []

for i in range(len(df)):

    pred_id = predictions[i].item()

    label = model.config.id2label[
        pred_id
    ].upper()

    confidence = float(
        probabilities[i][pred_id]
    )

    labels.append(label)
    confidences.append(confidence)

df["nli_label"] = labels
df["nli_confidence"] = confidences

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 65)
print("REAL-TIME NLI COMPLETE")
print("=" * 65)

for _, row in df.iterrows():

    print(
        f"\nRank: {row['final_rank']}"
    )

    print(
        f"Source: {row['website']}"
    )

    print(
        f"NLI: {row['nli_label']}"
    )

    print(
        f"Confidence: "
        f"{row['nli_confidence']:.4f}"
    )

print("\nSaved to:")
print(OUTPUT_FILE)