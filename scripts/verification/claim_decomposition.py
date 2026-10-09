import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def decompose_claim(claim):

    prompt = f"""
You are the claim decomposition module of a misinformation verification system.

Break the given factual claim into the smallest meaningful atomic claims
that can be independently verified using evidence.

Rules:
1. Return 1 to 5 atomic claims.
2. Each atomic claim must be independently verifiable.
3. Preserve important dates, places, people, organizations and numbers.
4. Do not add information that is not present in the original claim.
5. Do not change the meaning of the original claim.
6. If the claim is already simple, return it as ONE atomic claim.
7. Return ONLY a numbered list.
8. Do not create duplicate or overlapping claims.
9. Do not split the same fact into separate claims just because it contains
   related details such as location, time, or scope.
10. Combine related information when it can be verified as one factual statement.

Example:

Claim:
UPI is completely shut down across India from today.

Atomic claims:
1. UPI is completely shut down in India.
2. The shutdown started today.

Claim:
NASA confirmed that humans will live on Mars by 2030.

Atomic claims:
1. NASA confirmed that humans will live on Mars by 2030.

Now decompose this claim:

CLAIM:
{claim}

Atomic claims:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()


if __name__ == "__main__":

    print("=" * 70)
    print("TRUTHLENS - CLAIM DECOMPOSITION")
    print("=" * 70)

    claim = input("\nEnter extracted claim:\n> ")

    result = decompose_claim(claim)

    print("\nAtomic Claims:")
    print(result)