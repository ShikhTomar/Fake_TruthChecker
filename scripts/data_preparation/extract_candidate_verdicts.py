import pandas as pd
from pathlib import Path
import re


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
print("TRUTHLENS - CANDIDATE VERDICT EXTRACTION")
print("=" * 80)

print("Excel files found:", len(excel_files))


# ============================================================
# 2. VERDICT PATTERNS
# ============================================================
# IMPORTANT:
# These are ONLY candidate labels.
# They are NOT ground-truth labels.

patterns = {

    "CONTRADICTED": [

        # Explicit fact-check verdicts
        r"\bfact\s*check\s*[:\-]\s*(false|fake)\b",

        # Direct conclusion statements
        r"\bclaim\s+is\s+(false|fake|incorrect|untrue)\b",
        r"\bclaim\s+was\s+(false|fake|incorrect|untrue)\b",

        # Common fact-check wording
        r"\b(false|fake)\s+claim\b",
        r"\b(false|fake)\s+information\b",
        r"\b(false|fake)\s+news\b",
        r"\b(false|fake)\s+post\b",
        r"\b(false|fake)\s+message\b",

        # Strong conclusion phrases
        r"\bnot\s+true\b",
        r"\bdoes\s+not\s+show\b",
        r"\bdid\s+not\s+happen\b",
        r"\bnever\s+happened\b",
        r"\bno\s+evidence\b"
    ],


    "MISLEADING": [

        # Explicit verdict
        r"\bfact\s*check\s*[:\-]\s*misleading\b",

        # Direct conclusion
        r"\bclaim\s+is\s+misleading\b",
        r"\bclaim\s+was\s+misleading\b",

        # Common wording
        r"\bmisleading\s+claim\b",
        r"\bmisleading\s+post\b",
        r"\bmisleading\s+information\b",
        r"\bmisleading\s+message\b",
        r"\bmisleading\s+video\b"
    ],


    "SUPPORTED": [

        # Explicit verdict
        r"\bfact\s*check\s*[:\-]\s*(true|correct)\b",

        # Direct conclusion
        r"\bclaim\s+is\s+(true|correct|accurate)\b",
        r"\bclaim\s+was\s+(true|correct|accurate)\b",

        # Verification wording
        r"\bclaim\s+(has\s+been|was)\s+verified\b",
        r"\bverified\s+claim\b",
        r"\bconfirmed\s+claim\b",
        r"\bclaim\s+is\s+verified\b",
        r"\bclaim\s+is\s+accurate\b"
    ]
}


# Compile regex patterns

compiled_patterns = {}

for label, pattern_list in patterns.items():

    compiled_patterns[label] = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in pattern_list
    ]


# ============================================================
# 3. HELPER FUNCTION
# ============================================================

def detect_verdict(title, investigation):

    title = "" if pd.isna(title) else str(title)
    investigation = "" if pd.isna(investigation) else str(investigation)

    # Search both fields
    text = title + "\n" + investigation

    detected_labels = []

    for label, regex_list in compiled_patterns.items():

        for regex in regex_list:

            if regex.search(text):

                detected_labels.append(label)
                break


    # --------------------------------------------------------
    # No verdict detected
    # --------------------------------------------------------

    if len(detected_labels) == 0:

        return (
            "UNKNOWN",
            "none",
            "LOW"
        )


    # --------------------------------------------------------
    # Multiple contradictory labels detected
    # --------------------------------------------------------

    if len(detected_labels) > 1:

        return (
            "AMBIGUOUS",
            "multiple_patterns",
            "LOW"
        )


    # --------------------------------------------------------
    # Exactly one candidate verdict
    # --------------------------------------------------------

    label = detected_labels[0]

    return (
        label,
        "title+investigation_rule",
        "CANDIDATE"
    )


# ============================================================
# 4. READ ALL EXCEL FILES
# ============================================================

all_rows = []

for file_path in excel_files:

    print("\nReading:", file_path.name)

    try:

        df = pd.read_excel(file_path)

    except Exception as e:

        print("ERROR:", e)
        continue


    for _, row in df.iterrows():

        title = row.get("title", "")
        claim = row.get("claim", "")
        investigation = row.get("investigation", "")

        verdict, source, confidence = detect_verdict(
            title,
            investigation
        )


        all_rows.append({

            "source_file":
                file_path.name,

            "website":
                row.get("website", ""),

            "publish_date":
                row.get("publish_date", ""),

            "title":
                "" if pd.isna(title)
                else str(title),

            "claim":
                "" if pd.isna(claim)
                else str(claim),

            "investigation":
                "" if pd.isna(investigation)
                else str(investigation),

            "detected_verdict":
                verdict,

            "verdict_source":
                source,

            "confidence":
                confidence
        })


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

result_df = pd.DataFrame(all_rows)


print("\n")
print("=" * 80)
print("VERDICT EXTRACTION COMPLETE")
print("=" * 80)

print("Total rows:", len(result_df))


# ============================================================
# 6. SHOW VERDICT DISTRIBUTION
# ============================================================

print("\nCandidate verdict distribution:")

print(
    result_df["detected_verdict"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 7. SHOW EXAMPLES
# ============================================================

print("\n")
print("=" * 80)
print("SAMPLE CANDIDATE VERDICTS")
print("=" * 80)

sample_columns = [
    "source_file",
    "title",
    "detected_verdict",
    "verdict_source",
    "confidence"
]

print(
    result_df[
        sample_columns
    ]
    .head(30)
    .to_string(index=False, max_colwidth=100)
)


# ============================================================
# 8. SAVE OUTPUT
# ============================================================

output_file = (
    FACTDRILL_FOLDER.parent
    / "factdrill_candidate_verdicts.csv"
)

result_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


print("\n")
print("=" * 80)
print("OUTPUT SAVED")
print("=" * 80)

print(output_file)


# ============================================================
# 9. SAVE ONLY UNKNOWN / AMBIGUOUS ROWS
# ============================================================

review_df = result_df[
    result_df["detected_verdict"].isin(
        ["UNKNOWN", "AMBIGUOUS"]
    )
]

review_file = (
    FACTDRILL_FOLDER.parent
    / "factdrill_verdicts_for_review.csv"
)

review_df.to_csv(
    review_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nRows requiring review:", len(review_df))

print("Review file:")
print(review_file)


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)