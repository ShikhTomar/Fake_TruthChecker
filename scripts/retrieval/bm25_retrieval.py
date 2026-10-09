import pandas as pd
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


# ============================================================
# 1. PATHS
# ============================================================

TRUTHLENS_FOLDER = (
    Path.home()
    / "Downloads"
    / "TruthLens"
)

EVIDENCE_FILE = (
    TRUTHLENS_FOLDER
    / "factdrill_evidence_corpus.csv"
)

TRAIN_FILE = (
    TRUTHLENS_FOLDER
    / "splits"
    / "train.csv"
)

OUTPUT_FILE = (
    TRUTHLENS_FOLDER
    / "bm25_test_results.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

evidence_df = pd.read_csv(
    EVIDENCE_FILE,
    encoding="utf-8-sig"
)

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8-sig"
)

# Only 500 claims for the first test
train_df = train_df.head(500)

print("=" * 70)
print("TRUTHLENS - BM25 SANITY TEST")
print("=" * 70)

print("Evidence documents:", len(evidence_df))
print("Test claims:", len(train_df))


# ============================================================
# 3. TOKENIZER
# ============================================================

def tokenize(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    return re.findall(
        r"\w+",
        text,
        flags=re.UNICODE
    )


# ============================================================
# 4. BUILD BM25 INDEX
# ============================================================

print("\nTokenizing evidence corpus...")

corpus = (
    evidence_df["investigation"]
    .fillna("")
    .astype(str)
    .apply(tokenize)
    .tolist()
)

print("Building BM25 index...")

bm25 = BM25Okapi(corpus)

print("BM25 index ready.")


# ============================================================
# 5. RETRIEVE TOP 5
# ============================================================

TOP_K = 5

results = []

for counter, (_, row) in enumerate(
    train_df.iterrows(),
    start=1
):

    if counter % 50 == 0:
        print(
            f"Processed {counter}/{len(train_df)} claims"
        )

    query = tokenize(
        row["claim"]
    )

    scores = bm25.get_scores(
        query
    )

    top_indices = scores.argsort()[
        ::-1
    ][:TOP_K]

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        evidence_row = evidence_df.iloc[idx]

        results.append({

            "claim_id":
                row["id"],

            "claim":
                row["claim"],

            "rank":
                rank,

            "evidence_id":
                evidence_row["evidence_id"],

            "evidence":
                evidence_row["investigation"],

            "website":
                evidence_row["website"],

            "bm25_score":
                float(scores[idx])
        })


# ============================================================
# 6. SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. DISPLAY FIRST CLAIM
# ============================================================

print("\n" + "=" * 70)
print("FIRST CLAIM RETRIEVAL")
print("=" * 70)

first_claim = (
    results_df["claim_id"]
    .iloc[0]
)

example = results_df[
    results_df["claim_id"] == first_claim
]

print("\nCLAIM:")
print(example.iloc[0]["claim"])

for _, row in example.iterrows():

    print("\nRank:", row["rank"])
    print("Score:", round(row["bm25_score"], 3))
    print("Website:", row["website"])
    print(
        "Evidence:",
        row["evidence"][:500]
    )


# ============================================================
# 8. FINISH
# ============================================================

print("\n" + "=" * 70)
print("BM25 SANITY TEST COMPLETE")
print("=" * 70)

print("Results:", len(results_df))

print(
    "Saved:",
    OUTPUT_FILE
)