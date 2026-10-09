import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

HYBRID_FILE = BASE / "hybrid_retrieval_results.csv"
CLAIMS_FILE = BASE / "splits" / "train.csv"
EVIDENCE_FILE = BASE / "factdrill_evidence_corpus.csv"

OUTPUT_FILE = BASE / "nli_verification_results.csv"

MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--MoritzLaurer--mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    / "snapshots"
    / "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
)

# ============================================================
# LOAD FILES
# ============================================================

print("Loading hybrid retrieval results...")

hybrid = pd.read_csv(
    HYBRID_FILE,
    encoding="utf-8-sig"
)

print("Hybrid rows:", len(hybrid))

print("Loading claims...")

claims = pd.read_csv(
    CLAIMS_FILE,
    encoding="utf-8-sig"
)[["id", "claim"]]

claims["id"] = claims["id"].astype(str)

print("Loading evidence corpus...")

evidence = pd.read_csv(
    EVIDENCE_FILE,
    encoding="utf-8-sig"
)[["evidence_id", "investigation"]]

evidence["evidence_id"] = (
    pd.to_numeric(
        evidence["evidence_id"],
        errors="coerce"
    )
    .astype("Int64")
    .astype(str)
)

# ============================================================
# MERGE CLAIMS
# ============================================================

hybrid["id"] = hybrid["id"].astype(str)

data = hybrid.merge(
    claims,
    on="id",
    how="left"
).head(100)

print("Claims matched:", data["claim"].notna().sum())

# ============================================================
# EVIDENCE LOOKUP
# ============================================================

evidence_lookup = dict(
    zip(
        evidence["evidence_id"],
        evidence["investigation"]
    )
)

# ============================================================
# CREATE NLI PAIRS
# ============================================================

rows = []

for _, row in data.iterrows():

    claim_id = row["id"]
    claim = row["claim"]

    if pd.isna(claim):
        continue

    # Format:
    # 1|15117|21055|16832|12740
    evidence_ids = str(
        row["hybrid_top5"]
    ).split("|")

    for rank, evidence_id in enumerate(
        evidence_ids,
        start=1
    ):

        evidence_id = str(
            evidence_id
        ).strip()

        evidence_text = evidence_lookup.get(
            evidence_id
        )

        if pd.isna(evidence_text) or not evidence_text:
            continue

        rows.append({
            "claim_id": claim_id,
            "claim": claim,
            "evidence_id": evidence_id,
            "evidence": evidence_text,
            "retrieval_rank": rank
        })

nli_df = pd.DataFrame(rows)

print("\nNLI pairs created:", len(nli_df))

if len(nli_df) == 0:
    raise ValueError(
        "Still no NLI pairs were created. "
        "Check evidence_id format."
    )

# ============================================================
# LOAD LOCAL NLI MODEL
# ============================================================

print("\nLoading local NLI model...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
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
# NLI LABEL
# ============================================================

def clean_label(label):

    label = str(label).lower()

    if "entail" in label:
        return "ENTAILMENT"

    if "contrad" in label:
        return "CONTRADICTION"

    if "neutral" in label:
        return "NEUTRAL"

    return label.upper()

# ============================================================
# NLI INFERENCE
# ============================================================

BATCH_SIZE = 32

labels = []
confidences = []

print("\nRunning NLI...")

for start in tqdm(
    range(0, len(nli_df), BATCH_SIZE),
    desc="NLI batches"
):

    batch = nli_df.iloc[
        start:start + BATCH_SIZE
    ]

    claims_batch = (
        batch["claim"]
        .astype(str)
        .tolist()
    )

    evidence_batch = (
        batch["evidence"]
        .astype(str)
        .tolist()
    )

    # Evidence = premise
    # Claim = hypothesis

    inputs = tokenizer(
        evidence_batch,
        claims_batch,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

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

    for i in range(len(batch)):

        pred_id = predictions[i].item()

        label = clean_label(
            model.config.id2label[pred_id]
        )

        confidence = float(
            probabilities[i][pred_id]
        )

        labels.append(label)
        confidences.append(confidence)

# ============================================================
# SAVE
# ============================================================

nli_df["nli_label"] = labels
nli_df["nli_confidence"] = confidences

nli_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NLI VERIFICATION COMPLETE")
print("=" * 60)

print("Total NLI pairs:", len(nli_df))

print("\nNLI distribution:")
print(
    nli_df["nli_label"].value_counts()
)

print("\nAverage confidence:")
print(
    round(
        nli_df["nli_confidence"].mean(),
        4
    )
)

print("\nSaved to:")
print(OUTPUT_FILE)