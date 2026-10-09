import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

INPUT_FILE = BASE / "nli_enriched_results.csv"
OUTPUT_FILE = BASE / "final_verdicts.csv"

# ============================================================
# LOAD
# ============================================================

print("Loading enriched NLI results...")

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("Rows:", len(df))

# ============================================================
# WEIGHTED SUPPORT / CONTRADICTION
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
# AGGREGATE PER CLAIM
# ============================================================

def resolve_claim(group):

    support = group["support_weight"].sum()
    contradiction = group["contradiction_weight"].sum()

    neutral = (
        group["nli_label"] == "NEUTRAL"
    ).sum()

    entailment_count = (
        group["nli_label"] == "ENTAILMENT"
    ).sum()

    contradiction_count = (
        group["nli_label"] == "CONTRADICTION"
    ).sum()

    total_evidence = len(group)

    # --------------------------------------------------------
    # Evidence sufficiency
    # --------------------------------------------------------

    if total_evidence == 0:
        verdict = "INSUFFICIENT_EVIDENCE"

    elif (
        support == 0
        and contradiction == 0
    ):
        verdict = "INSUFFICIENT_EVIDENCE"

    else:

        # ----------------------------------------------------
        # Conflict resolution
        # ----------------------------------------------------

        if support > contradiction:
            verdict = "SUPPORTED"

        elif contradiction > support:
            verdict = "CONTRADICTED"

        else:
            verdict = "INSUFFICIENT_EVIDENCE"

    # --------------------------------------------------------
    # Dominant evidence
    # --------------------------------------------------------

    total_signal = support + contradiction

    if total_signal > 0:

        support_ratio = (
            support / total_signal
        )

    else:
        support_ratio = 0.0

    contradiction_ratio = (
        1 - support_ratio
        if total_signal > 0
        else 0.0
    )

    return pd.Series({

        "claim": group["claim"].iloc[0],

        "support_weight": support,

        "contradiction_weight": contradiction,

        "support_ratio": support_ratio,

        "contradiction_ratio": contradiction_ratio,

        "entailment_count": entailment_count,

        "contradiction_count": contradiction_count,

        "neutral_count": neutral,

        "evidence_count": total_evidence,

        "verdict": verdict
    })


# ============================================================
# RUN
# ============================================================

print("\nResolving evidence conflicts...")

results = (
    df
    .groupby("claim_id", as_index=False)
    .apply(
        resolve_claim,
        include_groups=False
    )
    .reset_index(drop=True)
)

# ============================================================
# VERDICT CONFIDENCE
# ============================================================

def calculate_confidence(row):

    signal = (
        row["support_weight"]
        + row["contradiction_weight"]
    )

    if signal == 0:
        return 0.0

    if row["verdict"] == "SUPPORTED":
        return (
            row["support_weight"] / signal
        )

    if row["verdict"] == "CONTRADICTED":
        return (
            row["contradiction_weight"] / signal
        )

    return 0.0


results["verdict_confidence"] = results.apply(
    calculate_confidence,
    axis=1
)

# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONFLICT RESOLUTION COMPLETE")
print("=" * 60)

print("\nFinal verdict distribution:")

print(
    results["verdict"].value_counts()
)

print(
    "\nAverage verdict confidence:",
    round(
        results["verdict_confidence"].mean(),
        4
    )
)

print("\nSaved to:")
print(OUTPUT_FILE)