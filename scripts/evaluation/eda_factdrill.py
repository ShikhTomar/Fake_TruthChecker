import pandas as pd
import matplotlib.pyplot as plt
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

EDA_FOLDER = (
    TRUTHLENS_FOLDER
    / "eda_results"
)

EDA_FOLDER.mkdir(
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("TRUTHLENS - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(df.info())


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage":
        (df.isna().mean() * 100).round(2)
})

print(missing)


# Save missing-value report

missing.to_csv(
    EDA_FOLDER / "missing_values.csv"
)


# ============================================================
# 5. SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SOURCE DISTRIBUTION")
print("=" * 70)

source_counts = (
    df["website"]
    .fillna("Unknown")
    .value_counts()
)

print(source_counts)

source_counts.to_csv(
    EDA_FOLDER / "source_distribution.csv"
)


# ============================================================
# 6. SOURCE FILE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SOURCE FILE DISTRIBUTION")
print("=" * 70)

file_counts = (
    df["source_file"]
    .value_counts()
)

print(file_counts)

file_counts.to_csv(
    EDA_FOLDER / "source_file_distribution.csv"
)


# ============================================================
# 7. TEXT LENGTH ANALYSIS
# ============================================================

text_columns = [
    "claim",
    "title",
    "investigation",
    "content"
]

for column in text_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )

    df[column + "_word_count"] = (
        df[column]
        .str.split()
        .str.len()
    )

    df[column + "_char_count"] = (
        df[column]
        .str.len()
    )


print("\n" + "=" * 70)
print("TEXT LENGTH STATISTICS")
print("=" * 70)

length_columns = []

for column in text_columns:

    length_columns.append(
        column + "_word_count"
    )

    length_columns.append(
        column + "_char_count"
    )

print(
    df[length_columns]
    .describe()
)


# Save statistics

df[length_columns].describe().to_csv(
    EDA_FOLDER / "text_length_statistics.csv"
)


# ============================================================
# 8. CLAIM LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["claim_word_count"],
    bins=50
)

plt.xlabel("Number of Words")
plt.ylabel("Number of Claims")
plt.title("Claim Length Distribution")

plt.tight_layout()

plt.savefig(
    EDA_FOLDER / "claim_length_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. INVESTIGATION LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["investigation_word_count"],
    bins=50
)

plt.xlabel("Number of Words")
plt.ylabel("Number of Investigations")
plt.title("Investigation Length Distribution")

plt.tight_layout()

plt.savefig(
    EDA_FOLDER / "investigation_length_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. TOPICS
# ============================================================

print("\n" + "=" * 70)
print("TOPICS")
print("=" * 70)

topic_counts = (
    df["topic"]
    .fillna("Unknown")
    .value_counts()
)

print(topic_counts.head(30))

topic_counts.head(30).to_csv(
    EDA_FOLDER / "topic_distribution.csv"
)


# ============================================================
# 11. TAGS
# ============================================================

print("\n" + "=" * 70)
print("TOP TAG VALUES")
print("=" * 70)

tag_counts = (
    df["tags"]
    .fillna("Unknown")
    .value_counts()
)

print(tag_counts.head(30))

tag_counts.head(30).to_csv(
    EDA_FOLDER / "tag_distribution.csv"
)


# ============================================================
# 12. DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)

dates = pd.to_datetime(
    df["publish_date"],
    errors="coerce"
)

print("Earliest:", dates.min())
print("Latest:", dates.max())


# ============================================================
# 13. LANGUAGE APPROXIMATION FROM SOURCE FILE
# ============================================================
# FactDrill filenames encode language for many files.
# We extract the final language token from the filename.
# This is only an exploratory approximation.

def extract_language(filename):

    name = str(filename).lower()

    known_languages = [
        "english",
        "hindi",
        "bangla",
        "assamese",
        "burmese",
        "gujrati",
        "malayalam",
        "marathi",
        "odia",
        "sinhala",
        "tamil",
        "telugu",
        "urdu",
        "punjabi"
    ]

    for language in known_languages:

        if language in name:
            return language

    return "unknown"


df["language_approx"] = (
    df["source_file"]
    .apply(extract_language)
)


print("\n" + "=" * 70)
print("APPROXIMATE LANGUAGE DISTRIBUTION")
print("=" * 70)

language_counts = (
    df["language_approx"]
    .value_counts()
)

print(language_counts)

language_counts.to_csv(
    EDA_FOLDER / "language_distribution.csv"
)


# ============================================================
# 14. CLAIM + INVESTIGATION AVAILABILITY
# ============================================================

print("\n" + "=" * 70)
print("CLAIM / INVESTIGATION AVAILABILITY")
print("=" * 70)

print(
    "Claims available:",
    df["claim"].str.strip().ne("").sum()
)

print(
    "Investigations available:",
    df["investigation"].str.strip().ne("").sum()
)

print(
    "Both available:",
    (
        df["claim"].str.strip().ne("")
        &
        df["investigation"].str.strip().ne("")
    ).sum()
)


# ============================================================
# 15. SAVE COMPLETE EDA DATA
# ============================================================

eda_file = (
    EDA_FOLDER
    / "factdrill_eda_with_lengths.csv"
)

df.to_csv(
    eda_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("EDA results saved in:")
print(EDA_FOLDER)

print("\nImportant files:")

print(" - missing_values.csv")
print(" - source_distribution.csv")
print(" - source_file_distribution.csv")
print(" - text_length_statistics.csv")
print(" - claim_length_distribution.png")
print(" - investigation_length_distribution.png")
print(" - topic_distribution.csv")
print(" - tag_distribution.csv")
print(" - language_distribution.csv")
print(" - factdrill_eda_with_lengths.csv")