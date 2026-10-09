import pandas as pd
import requests
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "Downloads" / "TruthLens"

VERDICT_FILE = BASE / "realtime_final_verdict.csv"
OUTPUT_FILE = BASE / "realtime_factcheck_result.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# ============================================================
# LOAD
# ============================================================

print("Loading real-time verdict...")

df = pd.read_csv(
    VERDICT_FILE,
    encoding="utf-8-sig"
)

claim = str(df["claim"].iloc[0])
verdict = "UNKNOWN"

# Recalculate from evidence-level signals
support = df["support_weight"].sum()
contradiction = df["contradiction_weight"].sum()

if support == 0 and contradiction == 0:
    verdict = "INSUFFICIENT_EVIDENCE"
elif support > contradiction:
    verdict = "SUPPORTED"
elif contradiction > support:
    verdict = "CONTRADICTED"
else:
    verdict = "INSUFFICIENT_EVIDENCE"

total_signal = support + contradiction

if total_signal > 0:
    if verdict == "SUPPORTED":
        confidence = support / total_signal
    elif verdict == "CONTRADICTED":
        confidence = contradiction / total_signal
    else:
        confidence = 0.0
else:
    confidence = 0.0

# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

evidence_context = ""

for i, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    evidence_context += f"""
Evidence {i}
Source: {row['website']}
URL: {row['url']}
Retrieval Rank: {row['final_rank']}
NLI Result: {row['nli_label']}
NLI Confidence: {row['nli_confidence']:.3f}

Article:
{row['evidence'][:5000]}

----------------------------------------
"""

# ============================================================
# RAG PROMPT
# ============================================================

prompt = f"""
You are the explanation component of a fact-checking system.

The system has already determined the verdict.
You must NOT change the verdict.

You must use ONLY the evidence provided below.

Do not use your own world knowledge.
Do not invent facts.
Do not introduce information that is not present in the evidence.

CLAIM:
{claim}

SYSTEM VERDICT:
{verdict}

SYSTEM CONFIDENCE:
{confidence:.3f}

RETRIEVED EVIDENCE:
{evidence_context}

Write a concise fact-check explanation.

Requirements:
1. Start with the verdict.
2. Explain why the system reached this verdict.
3. Refer to the strongest evidence.
4. Mention conflicting or neutral evidence if relevant.
5. Do not claim certainty beyond the evidence.
6. Keep it under 150 words.
"""

# ============================================================
# CALL OLLAMA
# ============================================================

print("\nGenerating explanation with Llama 3.2...")

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

explanation = result["response"].strip()

# ============================================================
# SAVE
# ============================================================

output = pd.DataFrame([
    {
        "claim": claim,
        "verdict": verdict,
        "verdict_confidence": confidence,
        "explanation": explanation
    }
])

output.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("TRUTHLENS REAL-TIME FACT CHECK")
print("=" * 70)

print("\nClaim:")
print(claim)

print("\nVerdict:")
print(verdict)

print(
    "\nConfidence:",
    round(confidence, 4)
)

print("\nExplanation:")
print(explanation)

print("\nSaved to:")
print(OUTPUT_FILE)