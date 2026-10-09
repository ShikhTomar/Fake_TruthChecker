import pandas as pd
from pathlib import Path

# ============================================================
# 1. FACTDRILL FOLDER
# ============================================================

FACTDRILL_FOLDER = (
    Path.home()
    / "Downloads"
    / "TruthLens"
    / "FactDrill"
)

excel_files = sorted(FACTDRILL_FOLDER.glob("*.xlsx"))

print("=" * 80)
print("TRUTHLENS - FACTDRILL CONTENT INSPECTION")
print("=" * 80)

print(f"Files found: {len(excel_files)}")


# ============================================================
# 2. READ 5 ROWS FROM EACH FILE
# ============================================================

samples = []

for file_path in excel_files:

    print(f"Reading: {file_path.name}")

    try:

        df = pd.read_excel(file_path)

        # Only keep columns useful for our TEXT-ONLY project
        useful_columns = [
            "website",
            "publish_date",
            "title",
            "claim",
            "investigation",
            "content",
            "topic",
            "tags"
        ]

        available_columns = [
            col for col in useful_columns
            if col in df.columns
        ]

        temp = df[available_columns].head(5).copy()

        # Keep track of original file
        temp["source_file"] = file_path.name

        samples.append(temp)

    except Exception as e:

        print(f"ERROR reading {file_path.name}: {e}")


# ============================================================
# 3. COMBINE ALL SAMPLES
# ============================================================

sample_df = pd.concat(
    samples,
    ignore_index=True
)

print("\n" + "=" * 80)
print("SAMPLE CREATED")
print("=" * 80)

print("Total sample rows:", len(sample_df))


# ============================================================
# 4. DISPLAY CLAIM + INVESTIGATION
# ============================================================

pd.set_option("display.max_colwidth", 500)

for index, row in sample_df.iterrows():

    print("\n" + "=" * 80)

    print("SOURCE FILE:")
    print(row.get("source_file", ""))

    print("\nWEBSITE:")
    print(row.get("website", ""))

    print("\nTITLE:")
    print(row.get("title", ""))

    print("\nCLAIM:")
    print(row.get("claim", ""))

    print("\nINVESTIGATION:")
    print(row.get("investigation", ""))

    print("\nCONTENT:")
    content = row.get("content", "")

    if pd.notna(content):
        print(str(content)[:1000])
    else:
        print("")

    print("=" * 80)


# ============================================================
# 5. SAVE SMALL SAMPLE FILE
# ============================================================

output_file = (
    FACTDRILL_FOLDER.parent
    / "factdrill_content_sample.xlsx"
)

sample_df.to_excel(
    output_file,
    index=False
)

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print("Small sample saved at:")
print(output_file)