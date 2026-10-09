import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

TRUTHLENS_FOLDER = (
    Path.home()
    / "Downloads"
    / "TruthLens"
)

TRAIN_FILE = (
    TRUTHLENS_FOLDER
    / "splits"
    / "train.csv"
)

EVIDENCE_FILE = (
    TRUTHLENS_FOLDER
    / "factdrill_evidence_corpus.csv"
)

OUTPUT_FILE = (
    TRUTHLENS_FOLDER
    / "bm25_evaluation_mapping.csv"
)


# ============================================================
# LOAD
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8-sig"
)

evidence_df = pd.read_csv(
    EVIDENCE_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("CREATING CLAIM → EVIDENCE EVALUATION MAPPING")
print("=" * 70)

print("Train claims:", len(train_df))
print("Evidence documents:", len(evidence_df))


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    return (
        str(text)
        .lower()
        .strip()
    )


train_df["investigation_key"] = (
    train_df["investigation"]
    .apply(normalize)
)

evidence_df["investigation_key"] = (
    evidence_df["investigation"]
    .apply(normalize)
)


# ============================================================
# MAP INVESTIGATION → EVIDENCE ID
# ============================================================

evidence_lookup = (
    evidence_df
    .drop_duplicates("investigation_key")
    .set_index("investigation_key")
    ["evidence_id"]
    .to_dict()
)


train_df["gold_evidence_id"] = (
    train_df["investigation_key"]
    .map(evidence_lookup)
)


# ============================================================
# CHECK MAPPING
# ============================================================

mapped = train_df[
    train_df["gold_evidence_id"].notna()
]

unmapped = train_df[
    train_df["gold_evidence_id"].isna()
]

print("\nMapped claims:", len(mapped))
print("Unmapped claims:", len(unmapped))


# ============================================================
# SAVE MAPPING
# ============================================================

mapping_df = mapped[
    [
        "id",
        "claim",
        "investigation",
        "gold_evidence_id"
    ]
].copy()

mapping_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("MAPPING COMPLETE")
print("=" * 70)

print(
    "Mapping coverage:",
    f"{len(mapped) / len(train_df) * 100:.2f}%"
)

print("\nSaved at:")
print(OUTPUT_FILE)