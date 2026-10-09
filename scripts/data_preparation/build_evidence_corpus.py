import pandas as pd
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
    / "factdrill_cleaned.csv"
)

OUTPUT_FILE = (
    TRUTHLENS_FOLDER
    / "factdrill_evidence_corpus.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("TRUTHLENS - BUILDING EVIDENCE CORPUS")
print("=" * 70)

print("Input rows:", len(df))


# ============================================================
# 3. KEEP ROWS WITH INVESTIGATION
# ============================================================

evidence_df = df[
    df["investigation"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
].copy()

print(
    "Rows with investigation:",
    len(evidence_df)
)


# ============================================================
# 4. SELECT EVIDENCE FIELDS
# ============================================================

evidence_df = evidence_df[
    [
        "source_file",
        "website",
        "publish_date",
        "title",
        "investigation",
        "topic",
        "tags"
    ]
].copy()


# ============================================================
# 5. CREATE EVIDENCE ID
# ============================================================

evidence_df.insert(
    0,
    "evidence_id",
    range(1, len(evidence_df) + 1)
)


# ============================================================
# 6. CLEAN INVESTIGATION TEXT
# ============================================================

evidence_df["investigation"] = (
    evidence_df["investigation"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# 7. REMOVE EXACT DUPLICATE EVIDENCE
# ============================================================

before = len(evidence_df)

evidence_df = evidence_df.drop_duplicates(
    subset=["investigation"]
).reset_index(drop=True)

print(
    "Duplicate investigations removed:",
    before - len(evidence_df)
)


# ============================================================
# 8. RECREATE EVIDENCE IDs
# ============================================================

evidence_df["evidence_id"] = range(
    1,
    len(evidence_df) + 1
)


# ============================================================
# 9. SAVE
# ============================================================

evidence_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("EVIDENCE CORPUS CREATED")
print("=" * 70)

print(
    "Final evidence documents:",
    len(evidence_df)
)

print("\nColumns:")

for column in evidence_df.columns:
    print(" -", column)

print("\nSaved at:")
print(OUTPUT_FILE)

print("=" * 70)