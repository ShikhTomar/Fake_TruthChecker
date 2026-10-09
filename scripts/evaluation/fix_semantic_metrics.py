import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Downloads" / "TruthLens"

DETAILS_FILE = BASE / "semantic_evaluation_details.csv"
OUTPUT_FILE = BASE / "semantic_evaluation_corrected.csv"

df = pd.read_csv(
    DETAILS_FILE,
    encoding="utf-8-sig"
)

# Normalize evidence IDs
def normalize_id(x):
    if pd.isna(x):
        return None

    try:
        return str(int(float(x)))
    except:
        return str(x).strip()


df["gold"] = df["gold_evidence_id"].apply(normalize_id)

def normalize_retrieved(x):
    if pd.isna(x):
        return []

    return [
        normalize_id(v)
        for v in str(x).split("|")
        if str(v).strip()
    ]


df["retrieved"] = df["retrieved_top5"].apply(normalize_retrieved)

# Find gold rank
def find_rank(row):
    for rank, evidence_id in enumerate(row["retrieved"], start=1):
        if evidence_id == row["gold"]:
            return rank
    return 0


df["gold_rank_corrected"] = df.apply(find_rank, axis=1)

N = len(df)

recall_1 = (df["gold_rank_corrected"] <= 1).sum() / N
recall_3 = (df["gold_rank_corrected"] <= 3).sum() / N
recall_5 = (df["gold_rank_corrected"] <= 5).sum() / N

# Queries where gold wasn't retrieved have rank 0
mrr = np.mean([
    1 / rank if rank > 0 else 0
    for rank in df["gold_rank_corrected"]
])

# Save corrected details
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("CORRECTED SEMANTIC RETRIEVAL RESULTS")
print("=" * 60)

print(f"Queries evaluated : {N}")
print(f"Recall@1          : {recall_1:.4f} ({recall_1*100:.2f}%)")
print(f"Recall@3          : {recall_3:.4f} ({recall_3*100:.2f}%)")
print(f"Recall@5          : {recall_5:.4f} ({recall_5*100:.2f}%)")
print(f"MRR               : {mrr:.4f}")

print("\nSaved:")
print(OUTPUT_FILE)