import pandas as pd
from pathlib import Path

BASE = Path.home() / "Downloads" / "TruthLens"

BM25_FILE = BASE / "bm25_test_results.csv"
SEMANTIC_FILE = BASE / "semantic_evaluation_corrected.csv"

OUTPUT_FILE = BASE / "hybrid_retrieval_results.csv"

RRF_K = 60
TOP_K = 5


# ============================================================
# LOAD
# ============================================================

bm25 = pd.read_csv(BM25_FILE, encoding="utf-8-sig")
semantic = pd.read_csv(SEMANTIC_FILE, encoding="utf-8-sig")
semantic = semantic.drop_duplicates(subset=["id"]).copy()
print("Unique semantic queries:", len(semantic))

print("BM25 rows:", len(bm25))
print("Semantic queries:", len(semantic))


# ============================================================
# NORMALIZE IDS
# ============================================================

def norm_id(x):
    try:
        return str(int(float(x)))
    except:
        return str(x).strip()


bm25["evidence_id"] = bm25["evidence_id"].apply(norm_id)
semantic["gold_evidence_id"] = semantic["gold_evidence_id"].apply(norm_id)


# ============================================================
# BUILD BM25 RANKINGS
# ============================================================

bm25_rankings = {}

for claim_id, group in bm25.groupby("claim_id"):

    group = group.sort_values("rank")

    bm25_rankings[claim_id] = [
        norm_id(x)
        for x in group["evidence_id"].head(20)
    ]


# ============================================================
# BUILD SEMANTIC RANKINGS
# ============================================================

semantic_rankings = {}

for _, row in semantic.iterrows():

    claim_id = row["id"]

    retrieved = str(
        row["retrieved_top5"]
    ).split("|")

    semantic_rankings[claim_id] = [
        norm_id(x)
        for x in retrieved
        if str(x).strip()
    ]


# ============================================================
# RRF FUSION
# ============================================================

results = []

for _, row in semantic.iterrows():

    claim_id = row["id"]
    gold = norm_id(row["gold_evidence_id"])

    bm25_list = bm25_rankings.get(claim_id, [])
    semantic_list = semantic_rankings.get(claim_id, [])

    scores = {}

    # BM25 contribution
    for rank, evidence_id in enumerate(bm25_list, start=1):

        scores[evidence_id] = (
            scores.get(evidence_id, 0)
            + 1 / (RRF_K + rank)
        )

    # Semantic contribution
    for rank, evidence_id in enumerate(semantic_list, start=1):

        scores[evidence_id] = (
            scores.get(evidence_id, 0)
            + 1 / (RRF_K + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_results = ranked[:TOP_K]

    retrieved_ids = [
        evidence_id
        for evidence_id, score in top_results
    ]

    gold_rank = 0

    for rank, evidence_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if evidence_id == gold:
            gold_rank = rank
            break

    results.append({
        "id": claim_id,
        "gold_evidence_id": gold,
        "hybrid_top5": "|".join(retrieved_ids),
        "gold_rank": gold_rank
    })


# ============================================================
# METRICS
# ============================================================

result_df = pd.DataFrame(results)

N = len(result_df)

recall_1 = (
    result_df["gold_rank"] <= 1
).sum() / N

recall_3 = (
    result_df["gold_rank"] <= 3
).sum() / N

recall_5 = (
    result_df["gold_rank"] <= 5
).sum() / N

mrr = sum(
    1 / r if r > 0 else 0
    for r in result_df["gold_rank"]
) / N


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 60)
print("HYBRID BM25 + SEMANTIC RESULTS")
print("=" * 60)

print(f"Queries evaluated : {N}")
print(f"Recall@1          : {recall_1:.4f} ({recall_1*100:.2f}%)")
print(f"Recall@3          : {recall_3:.4f} ({recall_3*100:.2f}%)")
print(f"Recall@5          : {recall_5:.4f} ({recall_5*100:.2f}%)")
print(f"MRR               : {mrr:.4f}")

print("\nSaved:")
print(OUTPUT_FILE)