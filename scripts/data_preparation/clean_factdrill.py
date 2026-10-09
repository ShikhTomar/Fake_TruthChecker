import pandas as pd
import re
import unicodedata
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
    / "factdrill_unified.csv"
)

OUTPUT_FILE = (
    TRUTHLENS_FOLDER
    / "factdrill_cleaned.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("TRUTHLENS - TEXT CLEANING")
print("=" * 70)

print("Rows:", len(df))


# ============================================================
# 3. TEXT COLUMNS
# ============================================================

text_columns = [
    "claim",
    "title",
    "investigation",
    "content"
]


# ============================================================
# 4. CLEANING FUNCTION
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Replace line breaks/tabs with spaces
    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text
    )

    # Normalize repeated whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ============================================================
# 5. APPLY CLEANING
# ============================================================

for column in text_columns:

    if column in df.columns:

        df[column] = df[column].apply(
            clean_text
        )


# ============================================================
# 6. REMOVE ROWS WITH EMPTY CLAIM
# ============================================================

before = len(df)

df = df[
    df["claim"].str.strip().ne("")
]

df = df.reset_index(
    drop=True
)

print(
    "Rows removed due to empty claim:",
    before - len(df)
)


# ============================================================
# 7. REMOVE DUPLICATES AFTER CLEANING
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=[
        "claim",
        "investigation"
    ]
)

df = df.reset_index(
    drop=True
)

print(
    "Duplicates removed:",
    before_duplicates - len(df)
)


# ============================================================
# 8. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("Final rows:", len(df))
print("Final columns:", len(df.columns))

print("\nSaved at:")
print(OUTPUT_FILE)

print("=" * 70)