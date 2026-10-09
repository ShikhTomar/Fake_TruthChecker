import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

NLI_FILE = BASE / "nli_verification_results.csv"
EVIDENCE_FILE = BASE / "factdrill_evidence_corpus.csv"

OUTPUT_FILE = BASE / "nli_enriched_results.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading NLI results...")

nli = pd.read_csv(
    NLI_FILE,
    encoding="utf-8-sig"
)

print("NLI rows:", len(nli))

print("Loading evidence metadata...")

evidence = pd.read_csv(
    EVIDENCE_FILE,
    encoding="utf-8-sig"
)

# Keep only required metadata
evidence_meta = evidence[
    [
        "evidence_id",
        "website",
        "publish_date",
        "title"
    ]
].copy()

# Normalize IDs
nli["evidence_id"] = (
    pd.to_numeric(
        nli["evidence_id"],
        errors="coerce"
    )
    .astype("Int64")
)

evidence_meta["evidence_id"] = (
    pd.to_numeric(
        evidence_meta["evidence_id"],
        errors="coerce"
    )
    .astype("Int64")
)

# ============================================================
# MERGE
# ============================================================

df = nli.merge(
    evidence_meta,
    on="evidence_id",
    how="left"
)

print(
    "Evidence metadata matched:",
    df["website"].notna().sum(),
    "/",
    len(df)
)

# ============================================================
# DATE PROCESSING
# ============================================================

df["publish_date"] = pd.to_datetime(
    df["publish_date"],
    errors="coerce"
)

# Latest evidence date in our current sample
reference_date = df["publish_date"].max()

print("\nReference date:", reference_date)

# Age in days
df["evidence_age_days"] = (
    reference_date - df["publish_date"]
).dt.days

# ============================================================
# RECENCY SCORE
# ============================================================

# More recent evidence gets a higher score.
#
# 0 days old  -> 1.0
# 365 days    -> ~0.37
# 730+ days   -> ~0.14
#
# This is an exponential decay, not a truth label.

df["recency_score"] = np.exp(
    -df["evidence_age_days"].fillna(3650) / 365
)

# ============================================================
# RETRIEVAL SCORE
# ============================================================

# Higher-ranked retrieval evidence gets higher weight.

df["retrieval_score"] = 1 / df["retrieval_rank"]

# ============================================================
# SOURCE CREDIBILITY
# ============================================================

# IMPORTANT:
# We are NOT claiming these sources are objectively truthful.
# This is only a configurable baseline source-priority signal.
#
# Unknown sources receive 1.0.

df["source_credibility"] = 1.0

# Fact-checking organizations in our corpus
# receive a modest baseline priority.

fact_check_domains = [
    "altnews",
    "boomlive",
    "factcrescendo",
    "factchecker",
    "factly",
    "indiatoday",
    "newschecker",
    "newsmobile",
    "vishwas",
    "webqoof"
]

for keyword in fact_check_domains:

    mask = (
        df["website"]
        .astype(str)
        .str.lower()
        .str.contains(
            keyword,
            na=False
        )
    )

    df.loc[
        mask,
        "source_credibility"
    ] = 1.2

# ============================================================
# COMBINED EVIDENCE WEIGHT
# ============================================================

df["evidence_weight"] = (
    df["nli_confidence"]
    * df["recency_score"]
    * df["retrieval_score"]
    * df["source_credibility"]
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL + CREDIBILITY PROCESSING COMPLETE")
print("=" * 60)

print("\nNLI distribution:")
print(
    df["nli_label"].value_counts()
)

print("\nAverage recency score:",
      round(df["recency_score"].mean(), 4))

print(
    "Average source credibility:",
    round(df["source_credibility"].mean(), 4)
)

print(
    "Average evidence weight:",
    round(df["evidence_weight"].mean(), 4)
)

print("\nSaved to:")
print(OUTPUT_FILE)