import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

INPUT_FILE = BASE / "realtime_nli_results.csv"
OUTPUT_FILE = BASE / "realtime_final_verdict.csv"

# ============================================================
# LOAD
# ============================================================

print("Loading real-time NLI results...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Evidence rows:", len(df))

# ============================================================
# SOURCE CREDIBILITY
# ============================================================

# Baseline source-priority signal.
# This is NOT a claim that a source is always truthful.

df["source_credibility"] = 1.0

high_priority_sources = [
    "economictimes",
    "livemint",
    "news18",
    "reuters",
    "pib",
    "rbi",
    "npci",
    "indiatoday"
]

for keyword in high_priority_sources:

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
# RETRIEVAL SCORE
# ============================================================

df["retrieval_score"] = (
    1 / df["final_rank"]
)

# ============================================================
# TEMPORAL SCORE
# ============================================================

# Current date is used only as the reference point.
# No publication date is invented.

today = pd.Timestamp(
    datetime.now().date()
)

df["temporal_score"] = 1.0

if "publish_date" in df.columns:

    dates = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )

    valid_dates = dates.notna()

    age_days = (
        today - dates
    ).dt.days

    df.loc[
        valid_dates,
        "temporal_score"
    ] = np.exp(
        -age_days[valid_dates] / 365
    )


# ============================================================
# EVIDENCE WEIGHT
# ============================================================

df["evidence_weight"] = (
    df["nli_confidence"]
    * df["retrieval_score"]
    * df["source_credibility"]
    * df["temporal_score"]
)

# ============================================================
# SUPPORT / CONTRADICTION
# ============================================================

df["support_weight"] = 0.0
df["contradiction_weight"] = 0.0

df.loc[
    df["nli_label"] == "ENTAILMENT",
    "support_weight"
] = df["evidence_weight"]

df.loc[
    df["nli_label"] == "CONTRADICTION",
    "contradiction_weight"
] = df["evidence_weight"]

# ============================================================
# AGGREGATE
# ============================================================

support = df["support_weight"].sum()

contradiction = df[
    "contradiction_weight"
].sum()

neutral = (
    df["nli_label"] == "NEUTRAL"
).sum()

# ============================================================
# FINAL VERDICT
# ============================================================

if support == 0 and contradiction == 0:

    verdict = "INSUFFICIENT_EVIDENCE"

elif support > contradiction:

    verdict = "SUPPORTED"

elif contradiction > support:

    verdict = "CONTRADICTED"

else:

    verdict = "INSUFFICIENT_EVIDENCE"


# ============================================================
# CONFIDENCE
# ============================================================

total_signal = (
    support + contradiction
)

if total_signal > 0:

    if verdict == "SUPPORTED":

        verdict_confidence = (
            support / total_signal
        )

    elif verdict == "CONTRADICTED":

        verdict_confidence = (
            contradiction / total_signal
        )

    else:

        verdict_confidence = 0.0

else:

    verdict_confidence = 0.0


# ============================================================
# SAVE EVIDENCE-LEVEL RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 65)
print("REAL-TIME CONFLICT RESOLUTION COMPLETE")
print("=" * 65)

print("\nClaim:")
print(df["claim"].iloc[0])

print("\nEvidence signals:")

print(
    "Support weight:",
    round(support, 4)
)

print(
    "Contradiction weight:",
    round(contradiction, 4)
)

print(
    "Neutral evidence:",
    neutral
)

print("\nFINAL VERDICT:")
print(verdict)

print(
    "\nVerdict confidence:",
    round(verdict_confidence, 4)
)

print("\nEvidence breakdown:")

print(
    df[
        [
            "website",
            "final_rank",
            "nli_label",
            "nli_confidence",
            "source_credibility",
            "evidence_weight"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)