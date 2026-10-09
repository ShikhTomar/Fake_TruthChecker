import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


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

OUTPUT_FOLDER = (
    TRUTHLENS_FOLDER
    / "splits"
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
print("TRUTHLENS - TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print("Total rows:", len(df))


# ============================================================
# 3. CREATE GROUP ID
# ============================================================
# Same normalized claim = same group.
# This prevents the same claim from appearing in
# different splits.

df["claim_group"] = (
    df["claim"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ============================================================
# 4. FIRST SPLIT
# ============================================================
# 70% TRAIN
# 30% TEMPORARY (validation + test)

gss_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    gss_1.split(
        df,
        groups=df["claim_group"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()


# ============================================================
# 5. SECOND SPLIT
# ============================================================
# Split the remaining 30% equally:
#
# 15% VALIDATION
# 15% TEST

gss_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    gss_2.split(
        temp_df,
        groups=temp_df["claim_group"]
    )
)

validation_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()


# ============================================================
# 6. REMOVE INTERNAL GROUP COLUMN
# ============================================================

train_df = train_df.drop(
    columns=["claim_group"]
)

validation_df = validation_df.drop(
    columns=["claim_group"]
)

test_df = test_df.drop(
    columns=["claim_group"]
)


# ============================================================
# 7. RESET INDEX
# ============================================================

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# 8. SAVE
# ============================================================

train_df.to_csv(
    OUTPUT_FOLDER / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    OUTPUT_FOLDER / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    OUTPUT_FOLDER / "test.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. VERIFY NO CLAIM LEAKAGE
# ============================================================

def normalize_claim(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"https?://\S+|www\.\S+",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


train_claims = set(
    normalize_claim(train_df["claim"])
)

validation_claims = set(
    normalize_claim(validation_df["claim"])
)

test_claims = set(
    normalize_claim(test_df["claim"])
)


train_val_overlap = (
    train_claims & validation_claims
)

train_test_overlap = (
    train_claims & test_claims
)

val_test_overlap = (
    validation_claims & test_claims
)


# ============================================================
# 10. REPORT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print(
    f"Train:       {len(train_df):,} "
    f"({len(train_df)/len(df)*100:.2f}%)"
)

print(
    f"Validation:  {len(validation_df):,} "
    f"({len(validation_df)/len(df)*100:.2f}%)"
)

print(
    f"Test:        {len(test_df):,} "
    f"({len(test_df)/len(df)*100:.2f}%)"
)

print("\nClaim overlap:")

print(
    "Train ↔ Validation:",
    len(train_val_overlap)
)

print(
    "Train ↔ Test:",
    len(train_test_overlap)
)

print(
    "Validation ↔ Test:",
    len(val_test_overlap)
)


# ============================================================
# 11. FINAL CHECK
# ============================================================

if (
    len(train_val_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(val_test_overlap) == 0
):

    print("\n✓ NO CLAIM LEAKAGE DETECTED")

else:

    print("\n⚠ CLAIM OVERLAP DETECTED")


print("\nFiles saved in:")
print(OUTPUT_FOLDER)

print("\nCreated:")
print(" - train.csv")
print(" - validation.csv")
print(" - test.csv")