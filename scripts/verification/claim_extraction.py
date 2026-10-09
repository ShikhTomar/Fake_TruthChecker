import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"


def extract_claim(social_media_text):

    prompt = f"""
You are a claim extraction component in a misinformation detection system.

Your task is to identify the MAIN FACTUAL ASSERTION in the social-media post.

A factual claim is a statement that can be checked as TRUE or FALSE using evidence.

IMPORTANT:
- If the post says something happened, is happening, or will happen, that is a claim.
- Statements about organizations, people, services, events, dates, numbers, or incidents can be claims.
- Ignore greetings and opinions.
- Ignore requests such as "share this", "like", or "follow".
- Do NOT return NO_CLAIM if the post contains a factual assertion.
- Do not invent information.
- Return exactly ONE claim.
- Return only the claim, with no explanation.

Examples:

Post:
"Breaking! UPI services are completely shut down across India today. Share this!"
Claim:
UPI services are completely shut down across India today.

Post:
"NASA confirmed that humans will live on Mars by 2030."
Claim:
NASA confirmed that humans will live on Mars by 2030.

Post:
"Please share this message with everyone."
Claim:
NO_CLAIM

Post:
"OMG this is terrible! I think the government is useless."
Claim:
NO_CLAIM

Now extract the claim from this post:

POST:
{social_media_text}

CLAIM:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.9
            }
        },
        timeout=120
    )

    response.raise_for_status()

    claim = response.json()["response"].strip()

    return claim


if __name__ == "__main__":

    print("=" * 70)
    print("TRUTHLENS - CLAIM EXTRACTION")
    print("=" * 70)

    social_media_text = input("\nEnter social-media post:\n> ")

    claim = extract_claim(social_media_text)

    print("\nExtracted Claim:")
    print(claim)