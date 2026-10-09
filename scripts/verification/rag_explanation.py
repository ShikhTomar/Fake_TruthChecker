import pandas as pd
import requests
import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

VERDICT_FILE = BASE / "final_verdicts.csv"
NLI_FILE = BASE / "nli_enriched_results.csv"

OUTPUT_FILE = BASE / "rag_explanations.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading final verdicts...")

verdicts = pd.read_csv(
    VERDICT_FILE,
    encoding="utf-8-sig"
)

print("Claims:", len(verdicts))

print("Loading evidence...")

evidence = pd.read_csv(
    NLI_FILE,
    encoding="utf-8-sig"
)

# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    claim,
    verdict,
    confidence,
    evidence_rows
):

    evidence_text = ""

    for i, (_, row) in enumerate(
        evidence_rows.iterrows(),
        start=1
    ):

        evidence_text += (
            f"\nEvidence {i}:\n"
            f"Source: {row['website']}\n"
            f"Date: {row['publish_date']}\n"
            f"NLI: {row['nli_label']}\n"
            f"Evidence: {row['evidence']}\n"
        )

    prompt = f"""
You are a fact-checking explanation system.

You MUST use ONLY the evidence provided below.

Do not use your own knowledge.
Do not invent facts.
Do not add information that is not present in the evidence.

Claim:
{claim}

System Verdict:
{verdict}

System Confidence:
{confidence:.2f}

Retrieved Evidence:
{evidence_text}

Write a short, clear fact-check explanation.

Rules:
1. Start with the verdict.
2. Explain why the system reached that verdict.
3. Mention the strongest evidence.
4. If evidence conflicts, mention the conflict.
5. Do not invent information.
6. Keep the explanation under 120 words.
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


# ============================================================
# PROCESS CLAIMS
# ============================================================

results = []

print("\nGenerating RAG explanations...")

for index, row in verdicts.iterrows():

    claim_id = row["claim_id"]

    claim_evidence = evidence[
        evidence["claim_id"].astype(str)
        == str(claim_id)
    ]

    # Use strongest 3 evidence pieces
    claim_evidence = (
        claim_evidence
        .sort_values(
            "evidence_weight",
            ascending=False
        )
        .head(3)
    )

    if len(claim_evidence) == 0:

        explanation = (
            "Insufficient evidence was available "
            "to generate a reliable explanation."
        )

    else:

        try:

            explanation = generate_explanation(
                row["claim"],
                row["verdict"],
                row["verdict_confidence"],
                claim_evidence
            )

        except Exception as e:

            explanation = (
                "Explanation generation failed: "
                + str(e)
            )

    results.append({
        "claim_id": claim_id,
        "claim": row["claim"],
        "verdict": row["verdict"],
        "verdict_confidence": row["verdict_confidence"],
        "explanation": explanation
    })

    print(
        f"[{index + 1}/{len(verdicts)}] "
        f"Claim {claim_id} processed"
    )


# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame(results)

output.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("RAG EXPLANATION COMPLETE")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_FILE)