import pandas as pd
import re
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

TRUTHLENS_FOLDER = (
    Path.home()
    / "Downloads"
    / "TruthLens"
)

INPUT_FILE = (
    TRUTHLENS_FOLDER
    / "eda_results"
    / "factdrill_eda_with_lengths.csv"
)

OUTPUT_FOLDER = (
    TRUTHLENS_FOLDER
    / "leakage_results"
)

OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("TRUTHLENS - DUPLICATE & DATA LEAKAGE CHECK")
print("=" * 70)

print("Total rows:", len(df))


# ============================================================
# 3. NORMALIZE CLAIMS
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["claim_normalized"] = (
    df["claim"]
    .apply(normalize_text)
)


# ============================================================
# 4. EXACT DUPLICATE CLAIMS
# ============================================================

print("\n" + "=" * 70)
print("EXACT DUPLICATE CLAIMS")
print("=" * 70)

claim_counts = (
    df["claim_normalized"]
    .value_counts()
)

duplicate_claims = claim_counts[
    claim_counts > 1
]

print(
    "Unique claims:",
    len(claim_counts)
)

print(
    "Duplicate claim groups:",
    len(duplicate_claims)
)

print(
    "Rows belonging to duplicate groups:",
    int(duplicate_claims.sum())
)


duplicate_df = df[
    df["claim_normalized"].isin(
        duplicate_claims.index
    )
].sort_values(
    "claim_normalized"
)

duplicate_df.to_csv(
    OUTPUT_FOLDER / "exact_duplicate_claims.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. CROSS-SOURCE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CROSS-SOURCE DUPLICATES")
print("=" * 70)

cross_source = (
    duplicate_df
    .groupby("claim_normalized")["website"]
    .nunique()
)

cross_source_claims = cross_source[
    cross_source > 1
]

print(
    "Claims appearing across multiple websites:",
    len(cross_source_claims)
)

cross_source_df = duplicate_df[
    duplicate_df["claim_normalized"].isin(
        cross_source_claims.index
    )
]

cross_source_df.to_csv(
    OUTPUT_FOLDER / "cross_source_duplicate_claims.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. SAME CLAIM / MULTIPLE INVESTIGATIONS
# ============================================================

print("\n" + "=" * 70)
print("SAME CLAIM / MULTIPLE INVESTIGATIONS")
print("=" * 70)

df["investigation_normalized"] = (
    df["investigation"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

investigation_counts = (
    df.groupby("claim_normalized")
    ["investigation_normalized"]
    .nunique()
)

multiple_investigations = (
    investigation_counts[
        investigation_counts > 1
    ]
)

print(
    "Claims with multiple investigations:",
    len(multiple_investigations)
)


# ============================================================
# 7. SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "unique_normalized_claims",
        "duplicate_claim_groups",
        "rows_in_duplicate_groups",
        "cross_source_duplicate_claims",
        "claims_with_multiple_investigations"
    ],

    "value": [
        len(df),
        len(claim_counts),
        len(duplicate_claims),
        int(duplicate_claims.sum()),
        len(cross_source_claims),
        len(multiple_investigations)
    ]
})


summary.to_csv(
    OUTPUT_FOLDER / "duplicate_leakage_summary.csv",
    index=False
)


# ============================================================
# 8. FINAL
# ============================================================

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)

print(
    "Results saved in:",
    OUTPUT_FOLDER
)

print(
    "\nUpload only:",
    OUTPUT_FOLDER / "duplicate_leakage_summary.csv"
)