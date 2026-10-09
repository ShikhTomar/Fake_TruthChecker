import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

INPUT_FILE = BASE / "realtime_evidence.csv"
OUTPUT_FILE = BASE / "realtime_ranked_evidence.csv"

MODEL_NAME = "intfloat/multilingual-e5-small"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading real-time evidence...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Evidence articles:", len(df))

claim = str(
    df["claim"].iloc[0]
)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading multilingual embedding model...")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("Device:", device)

# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating claim embedding...")

claim_embedding = model.encode(
    "query: " + claim,
    convert_to_tensor=True,
    normalize_embeddings=True
)

print("Creating evidence embeddings...")

evidence_texts = (
    "passage: "
    + df["title"].fillna("")
    + ". "
    + df["evidence"].fillna("").astype(str)
)

evidence_embeddings = model.encode(
    evidence_texts.tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    batch_size=16,
    show_progress_bar=True
)

# ============================================================
# SIMILARITY
# ============================================================

similarities = util.cos_sim(
    claim_embedding,
    evidence_embeddings
)[0]

df["semantic_score"] = (
    similarities
    .cpu()
    .numpy()
)

# ============================================================
# RANK
# ============================================================

df = df.sort_values(
    "semantic_score",
    ascending=False
).reset_index(drop=True)

df["final_rank"] = (
    range(1, len(df) + 1)
)

# ============================================================
# TOP 3
# ============================================================

top_k = 3

top = df.head(top_k)

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

print("\n" + "=" * 70)
print("REAL-TIME EVIDENCE RANKING COMPLETE")
print("=" * 70)

print("\nClaim:")
print(claim)

print("\nTop 3 evidence:")

for _, row in top.iterrows():

    print(
        f"\n[{row['final_rank']}] "
        f"Score: {row['semantic_score']:.4f}"
    )

    print(
        "Title:",
        row["title"]
    )

    print(
        "Website:",
        row["website"]
    )

    print(
        "URL:",
        row["url"]
    )

print("\nSaved to:")
print(OUTPUT_FILE)