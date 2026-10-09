import pandas as pd
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

FACTDRILL_FOLDER = (
    Path.home()
    / "Downloads"
    / "TruthLens"
    / "FactDrill"
)

OUTPUT_FILE = (
    FACTDRILL_FOLDER.parent
    / "factdrill_unified.csv"
)


# ============================================================
# 2. FIND ALL EXCEL FILES
# ============================================================

excel_files = sorted(
    FACTDRILL_FOLDER.glob("*.xlsx")
)

print("=" * 70)
print("TRUTHLENS - UNIFIED DATASET CREATION")
print("=" * 70)

print("Excel files found:", len(excel_files))


# ============================================================
# 3. STANDARD COLUMN NAMES
# ============================================================

column_mapping = {
    "linksInText": "links_in_text",
    "article_link": "link"
}


# ============================================================
# 4. COLUMNS WE WANT
# ============================================================

required_columns = [
    "website",
    "publish_date",
    "title",
    "claim",
    "investigation",
    "content",
    "topic",
    "tags"
]


# ============================================================
# 5. READ AND COMBINE
# ============================================================

all_data = []

for file_path in excel_files:

    print("Reading:", file_path.name)

    try:

        df = pd.read_excel(file_path)

        # Standardize known column-name differences
        df = df.rename(
            columns=column_mapping
        )

        # Add missing required columns
        for column in required_columns:

            if column not in df.columns:
                df[column] = pd.NA

        # Keep ONLY relevant text/metadata columns
        df = df[required_columns].copy()

        # Track original file
        df.insert(
            0,
            "source_file",
            file_path.name
        )

        all_data.append(df)

    except Exception as e:

        print(
            f"ERROR reading {file_path.name}: {e}"
        )


# ============================================================
# 6. COMBINE ALL FILES
# ============================================================

unified_df = pd.concat(
    all_data,
    ignore_index=True
)


print("\nTotal rows before cleaning:",
      len(unified_df))


# ============================================================
# 7. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

text_columns = [
    "title",
    "claim",
    "investigation",
    "content"
]

unified_df = unified_df.dropna(
    subset=text_columns,
    how="all"
)


# ============================================================
# 8. REMOVE ROWS WITHOUT CLAIM
# ============================================================

before_claim_removal = len(unified_df)

unified_df = unified_df[
    unified_df["claim"].notna()
]

unified_df = unified_df[
    unified_df["claim"]
    .astype(str)
    .str.strip()
    .ne("")
]


print(
    "Rows removed because claim was missing:",
    before_claim_removal - len(unified_df)
)


# ============================================================
# 9. NORMALIZE DATE
# ============================================================

unified_df["publish_date"] = pd.to_datetime(
    unified_df["publish_date"],
    errors="coerce"
)


# ============================================================
# 10. REMOVE EXACT DUPLICATE ROWS
# ============================================================

before_duplicates = len(unified_df)

unified_df = unified_df.drop_duplicates(
    subset=[
        "claim",
        "investigation"
    ]
)


print(
    "Exact claim+investigation duplicates removed:",
    before_duplicates - len(unified_df)
)


# ============================================================
# 11. RESET INDEX
# ============================================================

unified_df = unified_df.reset_index(
    drop=True
)


# ============================================================
# 12. CREATE INTERNAL ID
# ============================================================

unified_df.insert(
    0,
    "id",
    range(1, len(unified_df) + 1)
)


# ============================================================
# 13. SAVE
# ============================================================

unified_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("UNIFIED DATASET CREATED")
print("=" * 70)

print("Final rows:", len(unified_df))
print("Final columns:")

for column in unified_df.columns:
    print(" -", column)

print("\nSaved at:")
print(OUTPUT_FILE)