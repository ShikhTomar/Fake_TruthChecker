import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

INDEX_FILE = BASE / "semantic_index" / "evidence.faiss"
MAPPING_FILE = BASE / "semantic_index" / "evidence_mapping.csv"

BM25_FILE = BASE / "bm25_test_results.csv"
GOLD_FILE = BASE / "bm25_evaluation_mapping.csv"

OUTPUT_FILE = BASE / "semantic_evaluation_details.csv"


# ============================================================
# LOAD FILES
# ============================================================

print("Loading files...")

index = faiss.read_index(str(INDEX_FILE))

evidence = pd.read_csv(
    MAPPING_FILE,
    encoding="utf-8-sig"
)

bm25 = pd.read_csv(
    BM25_FILE,
    encoding="utf-8-sig"
)

gold = pd.read_csv(
    GOLD_FILE,
    encoding="utf-8-sig"
)

print("FAISS vectors:", index.ntotal)
print("BM25 test queries:", len(bm25))


# ============================================================
# LOAD MODEL
# ============================================================

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


# ============================================================
# GET CLAIMS
# ============================================================

# BM25 test file contains the claims
print("BM25 columns:", bm25.columns.tolist())

# Use the claim identifier available in the BM25 results
if "id" in bm25.columns:
    claims = bm25[["id", "claim"]].copy()

elif "claim_id" in bm25.columns:
    claims = bm25[["claim_id", "claim"]].rename(
        columns={"claim_id": "id"}
    )

else:
    # Fall back to matching using the claim text
    claims = bm25[["claim"]].copy()

    gold = gold[["id", "claim", "gold_evidence_id"]].copy()

    claims = claims.merge(
        gold[["id", "claim", "gold_evidence_id"]],
        on="claim",
        how="left"
    )

# Map claim → gold evidence
if "gold_evidence_id" not in claims.columns:
    claims = claims.merge(
        gold[["id", "gold_evidence_id"]],
        on="id",
        how="left"
    )

claims = claims.dropna(
    subset=["gold_evidence_id"]
).reset_index(drop=True)

claims["gold_evidence_id"] = (
    claims["gold_evidence_id"]
    .astype(str)
)

print("Queries with gold evidence:", len(claims))


# ============================================================
# CREATE CLAIM EMBEDDINGS
# ============================================================

query_texts = [
    "query: " + str(x)
    for x in claims["claim"]
]

print("\nGenerating claim embeddings...")

query_embeddings = model.encode(
    query_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")


# ============================================================
# SEMANTIC SEARCH
# ============================================================

TOP_K = 5

scores, indices = index.search(
    query_embeddings,
    TOP_K
)


# ============================================================
# EVALUATION
# ============================================================

evidence_ids = evidence["evidence_id"].astype(str).values

recall_1 = 0
recall_3 = 0
recall_5 = 0

reciprocal_ranks = []

details = []

for i in range(len(claims)):

    gold_id = claims.iloc[i]["gold_evidence_id"]

    retrieved = [
        str(evidence_ids[idx])
        for idx in indices[i]
        if idx >= 0
    ]

    rank = None

    for r, evidence_id in enumerate(retrieved, start=1):
        if evidence_id == gold_id:
            rank = r
            break

    if rank == 1:
        recall_1 += 1

    if rank is not None and rank <= 3:
        recall_3 += 1

    if rank is not None and rank <= 5:
        recall_5 += 1

    if rank is not None:
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

    details.append({
        "id": claims.iloc[i]["id"],
        "claim": claims.iloc[i]["claim"],
        "gold_evidence_id": gold_id,
        "retrieved_top5": "|".join(retrieved),
        "gold_rank": rank if rank is not None else 0
    })


# ============================================================
# METRICS
# ============================================================

N = len(claims)

recall_1 = recall_1 / N
recall_3 = recall_3 / N
recall_5 = recall_5 / N

mrr = np.mean(reciprocal_ranks)


# ============================================================
# SAVE DETAILS
# ============================================================

details_df = pd.DataFrame(details)

details_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SEMANTIC RETRIEVAL RESULTS")
print("=" * 60)

print(f"Queries evaluated : {N}")
print(f"Recall@1          : {recall_1:.4f} ({recall_1*100:.2f}%)")
print(f"Recall@3          : {recall_3:.4f} ({recall_3*100:.2f}%)")
print(f"Recall@5          : {recall_5:.4f} ({recall_5*100:.2f}%)")
print(f"MRR               : {mrr:.4f}")

print("\nSaved:")
print(OUTPUT_FILE)