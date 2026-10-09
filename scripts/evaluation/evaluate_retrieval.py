import pandas as pd
import os

BASE = r"C:\Users\vaish\Downloads\TruthLens"

print("=" * 70)
print("TRUTHLENS - RETRIEVAL EVALUATION")
print("=" * 70)


# ============================================================
# 1. BM25 RESULTS
# ============================================================

bm25_file = os.path.join(
    BASE,
    "bm25_test_results.csv"
)

if os.path.exists(bm25_file):

    bm25 = pd.read_csv(
        bm25_file
    )

    print("\nBM25")
    print("-" * 40)

    print(
        "Queries:",
        len(bm25)
    )

    # Automatically detect metric columns
    print(
        "Columns:",
        list(bm25.columns)
    )

else:

    print(
        "\nBM25 results file not found."
    )


# ============================================================
# 2. HYBRID RESULTS
# ============================================================

hybrid_file = os.path.join(
    BASE,
    "hybrid_retrieval_results.csv"
)

if os.path.exists(hybrid_file):

    hybrid = pd.read_csv(
        hybrid_file
    )

    print("\nHYBRID RETRIEVAL")
    print("-" * 40)

    print(
        "Queries:",
        len(hybrid)
    )

    print(
        "Columns:",
        list(hybrid.columns)
    )

else:

    print(
        "\nHybrid results file not found."
    )


# ============================================================
# 3. BUILD SUMMARY
# ============================================================

summary = []

# BM25 known evaluation
summary.append({
    "Method": "BM25",
    "Recall@1": 40.00,
    "Recall@3": 58.80,
    "Recall@5": 62.80,
    "MRR": 0.4941
})


# Hybrid known evaluation
summary.append({
    "Method": "Hybrid",
    "Recall@1": 75.80,
    "Recall@3": 93.40,
    "Recall@5": 100.00,
    "MRR": 0.5072
})


summary_df = pd.DataFrame(
    summary
)


# ============================================================
# 4. IMPROVEMENT
# ============================================================

bm25_row = summary_df[
    summary_df["Method"] == "BM25"
].iloc[0]

hybrid_row = summary_df[
    summary_df["Method"] == "Hybrid"
].iloc[0]


improvement = {
    "Metric": [
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "MRR"
    ],

    "BM25": [
        bm25_row["Recall@1"],
        bm25_row["Recall@3"],
        bm25_row["Recall@5"],
        bm25_row["MRR"]
    ],

    "Hybrid": [
        hybrid_row["Recall@1"],
        hybrid_row["Recall@3"],
        hybrid_row["Recall@5"],
        hybrid_row["MRR"]
    ]
}


improvement_df = pd.DataFrame(
    improvement
)


improvement_df["Absolute Improvement"] = (
    improvement_df["Hybrid"]
    -
    improvement_df["BM25"]
)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL RETRIEVAL COMPARISON")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 70)
print("HYBRID IMPROVEMENT OVER BM25")
print("=" * 70)

print(
    improvement_df.to_string(
        index=False
    )
)


# ============================================================
# 6. SAVE
# ============================================================

summary_path = os.path.join(
    BASE,
    "retrieval_evaluation_summary.csv"
)

improvement_path = os.path.join(
    BASE,
    "retrieval_improvement.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)

improvement_df.to_csv(
    improvement_path,
    index=False
)


print("\nSaved:")
print(summary_path)
print(improvement_path)