import os
import re
import math
import time
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import faiss
import torch

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ddgs import DDGS
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load GEMINI_API_KEY / GEMINI_MODEL from the .env file in this folder
load_dotenv(os.path.join(BASE_DIR, ".env"))

EVIDENCE_FILE = os.path.join(
    BASE_DIR,
    "factdrill_evidence_corpus.csv"
)

FAISS_INDEX_FILE = os.path.join(
    BASE_DIR,
    "semantic_index",
    "evidence.faiss"
)

FAISS_MAPPING_FILE = os.path.join(
    BASE_DIR,
    "semantic_index",
    "evidence_mapping.csv"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "truthlens_result.csv"
)

DETAIL_FILE = os.path.join(
    BASE_DIR,
    "truthlens_evidence_details.csv"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL") or None

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file next to truthlens.py "
        "containing:  GEMINI_API_KEY=your_key_here"
    )

llm_client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

NLI_MODEL = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

# Number of historical candidates retrieved by BM25
HISTORICAL_RETRIEVAL_K = 5

# Number of real-time search results requested
REALTIME_SEARCH_K = 10

# Maximum evidence considered after semantic ranking
FINAL_EVIDENCE_K = 6

# Number of evidence items sent to NLI
NLI_EVIDENCE_K = 3

# Minimum semantic similarity required for evidence
RELEVANCE_THRESHOLD = 0.70

# Fraction of the claim's content words that must appear (loosely stemmed)
# in the evidence passage. e5 similarity scores are compressed (unrelated text
# often scores 0.75-0.80), so the similarity threshold alone lets junk through.
MIN_CLAIM_TERM_OVERLAP = 0.50

# For current/date-sensitive claims, real-time evidence
# should be present before we confidently verify them.
CURRENT_CLAIM_REALTIME_MIN = 1

NLI_BATCH_SIZE = 8
MAX_NLI_LENGTH = 256


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

for required_path in (EVIDENCE_FILE, FAISS_INDEX_FILE, FAISS_MAPPING_FILE):
    if not os.path.isfile(required_path):
        raise FileNotFoundError(f"Required TruthLens file not found: {required_path}")

print("Loading historical evidence...")

evidence_df = pd.read_csv(
    EVIDENCE_FILE,
    low_memory=False
)

evidence_df["evidence_id"] = (
    evidence_df["evidence_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

evidence_df["investigation"] = (
    evidence_df["investigation"]
    .fillna("")
    .astype(str)
)

print(
    f"Historical evidence documents: "
    f"{len(evidence_df):,}"
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

faiss_index = faiss.read_index(
    FAISS_INDEX_FILE
)

faiss_mapping = pd.read_csv(
    FAISS_MAPPING_FILE,
    low_memory=False
)

faiss_mapping["evidence_id"] = (
    faiss_mapping["evidence_id"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

print(
    f"FAISS vectors loaded: "
    f"{faiss_index.ntotal:,}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# LOAD NLI MODEL
# ============================================================

print("Loading multilingual NLI model...")

nli_tokenizer = AutoTokenizer.from_pretrained(
    NLI_MODEL
)

nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL
)

nli_model.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model = nli_model.to(DEVICE)

# Read labels from the checkpoint itself; do not silently assume label order.
NLI_ID_TO_LABEL = {
    int(index): str(label).lower()
    for index, label in nli_model.config.id2label.items()
}
_REQUIRED_NLI_LABELS = {"entailment", "neutral", "contradiction"}
if set(NLI_ID_TO_LABEL.values()) != _REQUIRED_NLI_LABELS:
    raise RuntimeError(
        "Unexpected NLI model labels: "
        f"{NLI_ID_TO_LABEL}. Expected entailment, neutral, contradiction."
    )
NLI_LABEL_TO_ID = {label: index for index, label in NLI_ID_TO_LABEL.items()}

print("NLI labels:", NLI_ID_TO_LABEL)
print("Models loaded successfully.")


# ============================================================
# LLM HELPER (Gemini API)
# ============================================================

def call_llm(prompt, timeout=180):
    """Send a prompt to Gemini and return the text reply ("" on failure).

    - Retries when the server is busy (429 / 503 / 500).
    - If the main model keeps failing, switches to FALLBACK_MODEL (if set).
    - Permanent errors (wrong key, wrong model name) skip retries.
    """
    retryable_markers = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")
    models_to_try = [LLM_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])

    for model_name in models_to_try:
        for attempt in range(1, 6):
            try:
                response = llm_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return (response.text or "").strip()

            except Exception as e:
                message = str(e)
                print(f"LLM error ({model_name}, attempt {attempt}/5): {message[:150]}")

                if not any(marker in message for marker in retryable_markers):
                    break  # permanent error for this model, try the next one

                time.sleep(min(5 * attempt, 20))

    return ""


# Backward-compatible name: other files may still call ollama(...)
ollama = call_llm


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claim(text):

    prompt = f"""
You are the claim extraction component of a fact-checking system.

Your task is to extract the MAIN FACTUAL CLAIM from the given
social-media post.

Rules:

1. Return ONLY the factual claim.
2. Do not explain anything.
3. Do not answer the claim.
4. Do not say whether the claim is true or false.
5. Do not refuse the task.
6. Preserve important dates, locations, people, organizations,
   numbers and events.
7. If the post itself is already a factual claim, return it
   almost unchanged.
8. Ignore greetings, opinions and emotional language.

Examples:

Input:
"UPI is completely shut down in India from today."

Output:
UPI is completely shut down in India from today.

Input:
"Guys, is it true that India banned UPI today?"

Output:
India banned UPI today.

Input:
"Covid is back in 2026 in India"

Output:
Covid is back in 2026 in India.

Input:
"I heard that RBI has stopped issuing new notes."

Output:
RBI has stopped issuing new notes.

Now extract the main factual claim.

POST:
{text}

OUTPUT:
"""

    response = ollama(prompt)

    if not response:
        return text.strip()

    response = response.strip()

    response = re.sub(
        r"^(claim|output)\s*:\s*",
        "",
        response,
        flags=re.IGNORECASE
    ).strip().strip("\"'").strip()

    invalid_phrases = [
        "i can't assist",
        "i cannot assist",
        "i can't help",
        "i cannot help",
        "sorry, i",
        "unable to assist",
        "unable to help",
        "i don't know",
        "cannot provide",
        "i am unable",
        "i'm unable",
    ]

    if any(
        phrase in response.lower()
        for phrase in invalid_phrases
    ):
        return text.strip()

    return response


# ============================================================
# CLAIM DECOMPOSITION
# ============================================================

def decompose_claim(claim):

    # Conservative guard: a single-sentence input is already treated as
    # one claim. Do not let the LLM invent extra/background claims.
    # Multi-clause decomposition can be re-enabled later with a validator.
    claim = claim.strip()
    sentence_parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", claim) if x.strip()]
    if len(sentence_parts) <= 1:
        return [claim] if claim else []

    prompt = f"""
You are a claim decomposition module in a fact-checking system.

Your ONLY task is to split the given claim into atomic factual
claims.

CRITICAL RULES:

1. Use ONLY information explicitly contained in the original claim.
2. NEVER add new facts.
3. NEVER infer the opposite of the claim.
4. NEVER introduce "no evidence", "false", "true", "debunked",
   "fake", "misleading", or any other verification conclusion.
5. NEVER add information from your own knowledge.
6. NEVER turn an implied verification question into a new claim.
7. Do not explain anything.
8. Do not answer the claim.
9. If the original claim is already one atomic factual statement,
   return exactly ONE statement.
10. Preserve dates, locations, entities and events exactly.
11. Return at most 3 atomic claims.
12. Output ONLY the numbered claims.

Examples:

Original:
Covid is back in 2026 in India

Correct:
1. Covid is back in India in 2026.

WRONG:
1. Covid is back in India in 2026.
2. There is no evidence of Covid being back in India.

WRONG:
1. Covid cases have increased in India in 2026.

WRONG:
1. Covid is not back in India.

Original:
UPI is completely shut down in India from today.

Correct:
1. UPI is completely shut down in India.
2. The shutdown started today.

WRONG:
1. UPI is not shut down in India.

WRONG:
1. There is no evidence that UPI is shut down.

Original:
RBI banned 2000 rupee notes in Delhi yesterday.

Correct:
1. RBI banned 2000 rupee notes.
2. The ban applies in Delhi.
3. The ban happened yesterday.

Original claim:
{claim}

Return ONLY the numbered atomic factual claims.
"""

    response = ollama(prompt)

    if not response:
        return [claim.strip()]

    lines = response.splitlines()

    atomic_claims = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove numbering
        line = re.sub(
            r"^\s*(?:\d+[\.\)]|[-*])\s*",
            "",
            line
        ).strip()

        line = line.strip("\"'")

        if line:
            atomic_claims.append(line)

    # --------------------------------------------------------
    # INVALID RESPONSE FILTER
    # --------------------------------------------------------

    invalid_phrases = [
        "i can't assist",
        "i cannot assist",
        "i can't help",
        "i cannot help",
        "sorry",
        "unable to assist",
        "unable to help",
        "i don't know",
        "cannot provide",
        "i am unable",
        "i'm unable",
        "as an ai",
    ]

    valid_claims = []

    for c in atomic_claims:

        c_lower = c.lower()

        if any(
            phrase in c_lower
            for phrase in invalid_phrases
        ):
            continue

        if len(c.split()) < 3:
            continue

        valid_claims.append(c)

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_claims = []

    seen = set()

    for c in valid_claims:

        key = re.sub(
            r"\s+",
            " ",
            c.lower()
        ).strip()

        if key not in seen:

            seen.add(key)
            unique_claims.append(c)

    # --------------------------------------------------------
    # REMOVE VERIFICATION / OPPOSITE CLAIMS
    # --------------------------------------------------------

    verification_phrases = [
        "there is no evidence",
        "there's no evidence",
        "there is evidence",
        "there's evidence",
        "no evidence",
        "false",
        "true",
        "fake",
        "misleading",
        "debunked",
        "verified",
        "unverified",
        "not true",
        "isn't true",
        "is not true",
        "cannot be verified",
        "can't be verified",
        "unable to verify",
        "unable to confirm",
    ]

    filtered_claims = []

    for c in unique_claims:

        c_lower = c.lower()

        # Reject explicit verification conclusions
        if any(
            phrase in c_lower
            for phrase in verification_phrases
        ):
            continue

        # Reject assistant/refusal responses
        if any(
            phrase in c_lower
            for phrase in invalid_phrases
        ):
            continue

        filtered_claims.append(c)

    unique_claims = filtered_claims

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not unique_claims:
        return [claim.strip()]

    return unique_claims[:3]


# ============================================================
# HISTORICAL BM25 INDEX
# ============================================================

print("Building historical BM25 index...")

historical_documents = (
    evidence_df["investigation"]
    .fillna("")
    .astype(str)
    .tolist()
)

bm25_tokens = [
    doc.lower().split()
    for doc in historical_documents
]

bm25 = BM25Okapi(
    bm25_tokens
)


# ============================================================
# HISTORICAL RETRIEVAL
# ============================================================

def historical_retrieval(
    claim,
    top_k=HISTORICAL_RETRIEVAL_K
):

    query_tokens = claim.lower().split()

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        row = evidence_df.iloc[idx]

        results.append({
            "evidence_id": str(
                row["evidence_id"]
            ),
            "evidence": str(
                row["investigation"]
            ),
            "website": str(
                row.get(
                    "website",
                    ""
                )
            ),
            "title": str(
                row.get(
                    "title",
                    ""
                )
            ),
            "publish_date": str(
                row.get(
                    "publish_date",
                    ""
                )
            ),
            "topic": str(
                row.get(
                    "topic",
                    ""
                )
            ),
            "tags": str(
                row.get(
                    "tags",
                    ""
                )
            ),
            "retrieval_score": float(
                scores[idx]
            ),
            "source": "HISTORICAL"
        })

    return results


# ============================================================
# HTML CLEANING
# ============================================================

def clean_html(html):

    try:

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "aside"
            ]
        ):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text[:15000]

    except Exception:

        return ""


# ============================================================
# FETCH WEB PAGE
# ============================================================

def fetch_page(url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/154.0 Safari/537.36"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            return ""

        return clean_html(
            response.text
        )

    except Exception:

        return ""


# ============================================================
# REAL-TIME WEB RETRIEVAL
# ============================================================

def realtime_retrieval(
    claim,
    max_results=REALTIME_SEARCH_K
):

    results = []

    try:

        search_results = []
        for attempt in range(3):
            try:
                search_results = DDGS().text(
                    claim,
                    max_results=max_results
                )
                if search_results:
                    break
            except Exception as search_error:
                print(
                    f"Real-time search attempt {attempt + 1} failed:",
                    search_error
                )
            time.sleep(2)

        for item in search_results:

            title = item.get(
                "title",
                ""
            )

            url = item.get(
                "href",
                ""
            )

            snippet = item.get(
                "body",
                ""
            )

            page_text = ""

            if url:

                page_text = fetch_page(
                    url
                )

            if len(page_text) < 200:

                page_text = snippet

            if not page_text:
                continue

            results.append({
                "evidence_id":
                    f"RT_{len(results) + 1}",

                "evidence":
                    page_text,

                "website":
                    urlparse(
                        url
                    ).netloc
                    if url
                    else "",

                "title":
                    title,

                "publish_date":
                    "",

                "topic":
                    "",

                "tags":
                    "",

                "url":
                    url,

                "retrieval_score":
                    0.0,

                "source":
                    "REAL_TIME"
            })

    except Exception as e:

        print(
            "Real-time retrieval error:",
            e
        )

    return results


# ============================================================
# CURRENT / TIME-SENSITIVE CLAIM DETECTION
# ============================================================

def is_current_claim(claim):

    claim_lower = claim.lower()

    current_terms = [
        r"\btoday\b", r"\bnow\b", r"\bcurrently\b",
        r"\bcurrent\b", r"\blatest\b", r"\brecent(?:ly)?\b",
        r"\bthis year\b", r"\bthis month\b", r"\bthis week\b",
        r"\bat present\b", r"\bas of now\b",
        rf"\b{datetime.now().year}\b",
    ]

    return any(re.search(pattern, claim_lower) for pattern in current_terms)


# ============================================================
# RELEVANT PASSAGE EXTRACTION
# ============================================================

def extract_relevant_passage(claim, evidence_text, max_chars=1800):
    """Choose a compact sentence window relevant to the claim.

    This avoids feeding the NLI model only the top of a long webpage,
    which may contain navigation, introductions, or unrelated material.
    The full text remains available in the evidence-details CSV.
    """
    text = re.sub(r"\s+", " ", str(evidence_text or "")).strip()
    if len(text) <= max_chars:
        return text

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "of", "to", "in", "on", "at", "for", "and", "or",
        "but", "that", "this", "it", "as", "by", "with", "from",
        "around", "about", "into", "than", "then", "does", "do",
        "did", "has", "have", "had", "will", "would", "can", "could",
    }
    claim_terms = {
        token.lower() for token in re.findall(r"[\w'-]+", claim)
        if len(token) > 2 and token.lower() not in stop_words
    }
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if not sentences:
        return text[:max_chars]

    def score(sentence):
        terms = {token.lower() for token in re.findall(r"[\w'-]+", sentence)}
        return len(claim_terms & terms)

    best_index = max(range(len(sentences)), key=lambda i: score(sentences[i]))
    # Grow around the most overlapping sentence while respecting the size limit.
    selected = [sentences[best_index]]
    left, right = best_index - 1, best_index + 1
    while True:
        options = []
        if left >= 0:
            options.append((score(sentences[left]), left, "left"))
        if right < len(sentences):
            options.append((score(sentences[right]), right, "right"))
        if not options:
            break
        options.sort(reverse=True)
        _, index, side = options[0]
        candidate = ([sentences[index]] + selected) if side == "left" else (selected + [sentences[index]])
        if len(" ".join(candidate)) > max_chars:
            # Try the other side before stopping.
            options = options[1:]
            added = False
            for _, index2, side2 in options:
                candidate2 = ([sentences[index2]] + selected) if side2 == "left" else (selected + [sentences[index2]])
                if len(" ".join(candidate2)) <= max_chars:
                    selected = candidate2
                    if side2 == "left": left -= 1
                    else: right += 1
                    added = True
                    break
            if not added:
                break
        else:
            selected = candidate
            if side == "left": left -= 1
            else: right += 1
    passage = " ".join(selected)
    return passage[:max_chars]


# ============================================================
# LEXICAL RELEVANCE CHECK
# ============================================================

_OVERLAP_STOP_WORDS = {
    "the", "and", "for", "are", "was", "were", "has", "have", "had", "that",
    "this", "with", "from", "into", "than", "then", "does", "did", "will",
    "would", "can", "could", "about", "around", "its", "not", "but", "you",
    "your", "they", "their", "been", "being", "which", "who", "whom",
}


def claim_term_overlap(claim, evidence_text):
    """Fraction of the claim's content words found in the evidence passage.

    Words are matched on a 5-letter prefix so that freezes/freezing/freeze
    or revolves/revolve count as the same word.
    """
    def stems(text):
        return {
            token.lower()[:5]
            for token in re.findall(r"[A-Za-z]+", str(text))
            if len(token) > 2 and token.lower() not in _OVERLAP_STOP_WORDS
        }

    claim_stems = stems(claim)
    if not claim_stems:
        return 1.0
    passage = extract_relevant_passage(claim, evidence_text, max_chars=1800)
    return len(claim_stems & stems(passage)) / len(claim_stems)


# ============================================================
# E5 EVIDENCE RANKING
# ============================================================

def rank_evidence(
    claim,
    historical_results,
    realtime_results
):

    all_results = (
        historical_results +
        realtime_results
    )

    if not all_results:
        return []

    query_text = (
        "query: " +
        claim
    )

    query_embedding = embedding_model.encode(
        [query_text],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    evidence_texts = []

    for item in all_results:

        evidence_texts.append(
            "passage: " +
            extract_relevant_passage(claim, item.get("evidence", ""), max_chars=1800)
        )

    evidence_embeddings = (
        embedding_model.encode(
            evidence_texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
    )

    evidence_embeddings = np.asarray(
        evidence_embeddings,
        dtype="float32"
    )

    similarity_scores = (
        evidence_embeddings @
        query_embedding.T
    ).flatten()

    for item, score in zip(
        all_results,
        similarity_scores
    ):

        item["retrieval_score"] = float(
            score
        )

    # --------------------------------------------------------
    # Rank within each evidence source separately
    # --------------------------------------------------------

    historical_ranked = sorted(
        [
            x for x in all_results
            if x["source"] == "HISTORICAL"
        ],
        key=lambda x:
        x["retrieval_score"],
        reverse=True
    )

    realtime_ranked = sorted(
        [
            x for x in all_results
            if x["source"] == "REAL_TIME"
        ],
        key=lambda x:
        x["retrieval_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Preserve both branches
    # --------------------------------------------------------

    selected = []

    selected.extend(
        historical_ranked[
            :HISTORICAL_RETRIEVAL_K
        ]
    )

    selected.extend(
        realtime_ranked[
            :REALTIME_SEARCH_K
        ]
    )

    # Global ranking after preserving both branches
    selected.sort(
        key=lambda x:
        x["retrieval_score"],
        reverse=True
    )

    return selected[
        :FINAL_EVIDENCE_K
    ]


# ============================================================
# SOURCE CREDIBILITY
# ============================================================

def source_credibility(
    website
):

    website = str(
        website
    ).lower()

    high_credibility_domains = [
        "reuters",
        "pib",
        "rbi",
        "npci",
        "economictimes",
        "livemint",
        "news18",
        "indiatoday",
        "thehindu",
        "indianexpress",
        "timesofindia",
        "ndtv"
    ]

    for domain in high_credibility_domains:

        if domain in website:
            return 1.15

    return 1.00


# ============================================================
# TEMPORAL SCORE
# ============================================================

def temporal_score(
    publish_date
):

    if not publish_date:
        return 0.5

    try:

        date_text = str(
            publish_date
        ).strip()

        if not date_text:
            return 0.5

        parsed_date = pd.to_datetime(
            date_text,
            errors="coerce"
        )

        if pd.isna(parsed_date):
            return 0.5

        if parsed_date.tzinfo is not None:

            parsed_date = (
                parsed_date.tz_localize(None)
            )

        current_date = datetime.now()

        age_days = (
            current_date -
            parsed_date.to_pydatetime()
        ).days

        if age_days < 0:
            age_days = 0

        score = math.exp(
            -age_days / 180
        )

        return float(
            max(
                0.10,
                min(
                    1.0,
                    score
                )
            )
        )

    except Exception:

        return 0.5


# ============================================================
# EVIDENCE WEIGHT
# ============================================================

def evidence_weight(
    retrieval_score,
    nli_confidence,
    temporal,
    credibility
):

    relevance_factor = max(
        0.0,
        min(
            1.0,
            (retrieval_score - 0.70) / 0.30
        )
    )

    weight = (
        relevance_factor *
        nli_confidence *
        temporal *
        credibility
    )

    return float(
        max(
            0.0,
            weight
        )
    )


# ============================================================
# NLI
# ============================================================

# Directional (asymmetric) relations. NLI models rely heavily on word overlap,
# so "Earth revolves around the Sun" is often scored as ENTAILING
# "The Sun revolves around Earth". For these relations, swapping the two
# arguments changes the meaning, so we check the evidence for the swapped form.
DIRECTIONAL_RELATION_REGEX = (
    r"revolv(?:es|ed|e|ing) around|orbit(?:s|ed|ing)?|"
    r"rotat(?:es|ed|e|ing) around|goe?s around|went around|"
    r"travel(?:s|ed|ing)? around|circl(?:es|ed|e|ing)|"
    r"is (?:bigger|larger|smaller|heavier|lighter|older|younger|taller|"
    r"hotter|colder|faster|slower|greater|less) than|"
    r"defeated|beat|invaded|acquired|founded|invented|discovered|"
    r"killed|succeeded|replaced|is the capital of|is the parent of"
)

_ARTICLE_RE = re.compile(r"\b(?:the|a|an)\b\s*", flags=re.IGNORECASE)


def _strip_articles(text):
    return re.sub(r"\s+", " ", _ARTICLE_RE.sub("", str(text))).strip().lower()


def parse_directional_claim(claim):
    """Return (subject, relation, object) for 'A <directional relation> B' claims."""
    text = str(claim).strip().rstrip(".!?").strip()
    match = re.match(
        rf"^(?P<a>.+?)\s+(?P<rel>{DIRECTIONAL_RELATION_REGEX})\s+(?P<b>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    a = match.group("a").strip()
    rel = match.group("rel").strip()
    b = match.group("b").strip()
    if not a or not b:
        return None
    return a, rel, b


def swap_directional_claim(parsed):
    """Build the claim with subject and object exchanged."""
    a, rel, b = parsed
    a_part = re.sub(r"^The\b", "the", a)
    b_part = re.sub(r"^the\b", "The", b)
    b_part = b_part[0].upper() + b_part[1:] if b_part else b_part
    return f"{b_part} {rel} {a_part}."


def _literal_order_present(premise, subject, obj):
    """True if the premise literally says '<subject> <directional relation> <obj>'."""
    text = _strip_articles(premise)
    pattern = (
        r"\b" + re.escape(_strip_articles(subject))
        + r"\W+(?:" + DIRECTIONAL_RELATION_REGEX + r")\W+"
        + re.escape(_strip_articles(obj)) + r"\b"
    )
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _nli_probs(premises, hypotheses):
    inputs = nli_tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=MAX_NLI_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        return torch.softmax(nli_model(**inputs).logits, dim=-1).detach().cpu().numpy()


def nli(claim, evidence_list):
    """Run the trained multilingual NLI model directly.

    Do not turn every disagreement with a small generative model into NEUTRAL:
    that previous fallback erased all NLI signal and made decisive verdicts
    impossible. The model's class IDs for this checkpoint are 0=entailment,
    1=neutral, 2=contradiction.

    Role-reversal guard: for directional claims ("A revolves around B"),
    the evidence is also checked against the swapped claim ("B revolves
    around A"). If the evidence states the swapped version and not the
    original, the claim is CONTRADICTED, whatever the raw NLI label said.
    """
    if not evidence_list:
        return []

    premises = [
        extract_relevant_passage(claim, item.get("evidence", ""), max_chars=1800)
        for item in evidence_list
    ]
    hypotheses = [str(claim)] * len(premises)
    probs_batch = _nli_probs(premises, hypotheses)

    parsed = parse_directional_claim(claim)
    swapped_probs_batch = None
    if parsed:
        swapped_claim = swap_directional_claim(parsed)
        swapped_probs_batch = _nli_probs(premises, [swapped_claim] * len(premises))

    ent_id = NLI_LABEL_TO_ID["entailment"]
    neu_id = NLI_LABEL_TO_ID["neutral"]
    con_id = NLI_LABEL_TO_ID["contradiction"]

    id_to_label = NLI_ID_TO_LABEL
    results = []
    for i, (item, probs) in enumerate(zip(evidence_list, probs_batch)):
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        label = id_to_label.get(label_id, "neutral")

        role_reversal = False
        swapped_entailment = 0.0

        if parsed:
            swapped_entailment = float(swapped_probs_batch[i][ent_id])
            a, _rel, b = parsed
            says_original = _literal_order_present(premises[i], a, b)
            says_swapped = _literal_order_present(premises[i], b, a)

            nli_prefers_swapped = (
                swapped_entailment >= 0.50
                and swapped_entailment - float(probs[ent_id]) >= 0.10
            )

            if not says_original and (says_swapped or nli_prefers_swapped):
                role_reversal = True
                label = "contradiction"
                confidence = max(
                    swapped_entailment,
                    float(probs[con_id]),
                )

        results.append({
            "label": label,
            "confidence": confidence,
            "nli_model_label": id_to_label.get(label_id, "neutral"),
            "nli_model_confidence": float(probs[label_id]),
            "entailment_probability": float(probs[ent_id]),
            "neutral_probability": float(probs[neu_id]),
            "contradiction_probability": float(probs[con_id]),
            "swapped_entailment_probability": swapped_entailment,
            "role_reversal": role_reversal,
            "secondary_judge_label": "not_used",
        })
    return results


# ============================================================
# RESOLVE ONE ATOMIC CLAIM
# ============================================================

def resolve_atomic_claim(
    claim,
    ranked_evidence,
    current_claim=False
):

    if not ranked_evidence:

        return {
            "verdict":
                "INSUFFICIENT_EVIDENCE",

            "confidence":
                None,

            "evidence":
                []
        }

    # --------------------------------------------------------
    # Relevance filtering
    # --------------------------------------------------------

    relevant_evidence = [
        item
        for item in ranked_evidence
        if item["retrieval_score"]
        >= RELEVANCE_THRESHOLD
        and claim_term_overlap(
            claim,
            item.get("evidence", "")
        ) >= MIN_CLAIM_TERM_OVERLAP
    ]

    # --------------------------------------------------------
    # For current/date-sensitive claims:
    # historical evidence alone is insufficient.
    # --------------------------------------------------------

    realtime_relevant = [
        item
        for item in relevant_evidence
        if item["source"] == "REAL_TIME"
    ]

    if current_claim:

        if len(realtime_relevant) < (
            CURRENT_CLAIM_REALTIME_MIN
        ):

            return {
                "verdict":
                    "INSUFFICIENT_EVIDENCE",

                "confidence":
                    None,

                "evidence":
                    relevant_evidence[:FINAL_EVIDENCE_K]
            }

    # --------------------------------------------------------
    # Select evidence for NLI
    # --------------------------------------------------------

    if current_claim:

        # Prefer current web evidence.
        selected = (
            realtime_relevant[:NLI_EVIDENCE_K]
        )

    else:

        selected = (
            relevant_evidence[:NLI_EVIDENCE_K]
        )

    if not selected:

        return {
            "verdict":
                "INSUFFICIENT_EVIDENCE",

            "confidence":
                None,

            "evidence":
                ranked_evidence[:FINAL_EVIDENCE_K]
        }

    # --------------------------------------------------------
    # NLI
    # --------------------------------------------------------

    nli_results = nli(
        claim,
        selected
    )

    support_score = 0.0
    contradiction_score = 0.0
    total_evidence_weight = 0.0

    detailed_evidence = []

    for item, nli_result in zip(
        selected,
        nli_results
    ):

        label = nli_result[
            "label"
        ]

        confidence = nli_result[
            "confidence"
        ]

        retrieval_score = item[
            "retrieval_score"
        ]

        temporal = temporal_score(
            item.get(
                "publish_date",
                ""
            )
        )

        credibility = source_credibility(
            item.get(
                "website",
                ""
            )
        )

        # ----------------------------------------------------
        # Relevance gate
        # ----------------------------------------------------

        if (
            retrieval_score
            < RELEVANCE_THRESHOLD
        ):

            label = "neutral"
            confidence = 0.0

        weight = evidence_weight(
            retrieval_score,
            confidence,
            temporal,
            credibility
        )
        total_evidence_weight += weight

        if label == "entailment":

            support_score += weight

        elif label == "contradiction":

            contradiction_score += weight

        item["nli_label"] = label
        item["nli_passage"] = extract_relevant_passage(
            claim, item.get("evidence", ""), max_chars=1800
        )

        item["nli_confidence"] = confidence
        item["role_reversal"] = bool(nli_result.get("role_reversal", False))
        if "nli_model_label" in nli_result:
            item["nli_model_label"] = nli_result["nli_model_label"]
            item["nli_model_confidence"] = nli_result["nli_model_confidence"]
            item["secondary_judge_label"] = nli_result["secondary_judge_label"]

        item["temporal_score"] = (
            temporal
        )

        item["credibility"] = (
            credibility
        )

        item["evidence_weight"] = (
            weight
        )

        item["relevance_gate"] = (
            "ACCEPTED"
            if retrieval_score
            >= RELEVANCE_THRESHOLD
            else "REJECTED"
        )

        detailed_evidence.append(
            item
        )

    # --------------------------------------------------------
    # Verdict logic
    # --------------------------------------------------------

    total_signal = (
        support_score +
        contradiction_score
    )

    if total_signal <= 0:

        verdict = (
            "INSUFFICIENT_EVIDENCE"
        )

        confidence = None

    elif (
        contradiction_score
        > support_score
    ):

        verdict = "CONTRADICTED"

        confidence = (
            contradiction_score / total_evidence_weight
            if total_evidence_weight > 0 else None
        )

    elif (
        support_score
        > contradiction_score
    ):

        verdict = "SUPPORTED"

        confidence = (
            support_score / total_evidence_weight
            if total_evidence_weight > 0 else None
        )

    else:

        verdict = (
            "INSUFFICIENT_EVIDENCE"
        )

        confidence = None

    return {
        "verdict":
            verdict,

        "confidence":
            None if confidence is None else float(confidence),

        "evidence":
            detailed_evidence
    }


# ============================================================
# FINAL VERDICT ACROSS ATOMIC CLAIMS
# ============================================================

def final_verdict(
    atomic_results
):

    if not atomic_results:

        return (
            "INSUFFICIENT_EVIDENCE",
            None
        )

    verdicts = [
        r["verdict"]
        for r in atomic_results
    ]

    confidences = [
        r["confidence"]
        for r in atomic_results
    ]

    # --------------------------------------------------------
    # ALL SUPPORTED
    # --------------------------------------------------------

    if all(
        v == "SUPPORTED"
        for v in verdicts
    ):

        return (
            "SUPPORTED",
            float(min(c for c in confidences if c is not None))
        )

    # --------------------------------------------------------
    # ALL CONTRADICTED
    # --------------------------------------------------------

    if all(
        v == "CONTRADICTED"
        for v in verdicts
    ):

        return (
            "CONTRADICTED",
            float(min(c for c in confidences if c is not None))
        )

    # --------------------------------------------------------
    # MIXED / INSUFFICIENT
    #
    # Example:
    # SUPPORTED + INSUFFICIENT
    # CONTRADICTED + INSUFFICIENT
    # SUPPORTED + CONTRADICTED
    #
    # We do NOT allow one atomic claim to override
    # unresolved claims.
    # --------------------------------------------------------

    return (
        "INSUFFICIENT_EVIDENCE",
        None
    )


# ============================================================
# RAG EXPLANATION
# ============================================================

def format_confidence(value):
    """Display unknown confidence honestly instead of as 0%."""
    if value is None:
        return "Not established"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "Not established"


def generate_explanation(
    original_claim,
    final_verdict_text,
    final_confidence,
    all_evidence
):
    """Generate an explanation consistent with the final verdict.

    Abstentions use a deterministic explanation. The LLM is only asked to
    explain decisive verdicts, and is explicitly told not to change them.
    """
    usable_evidence = [
        item for item in all_evidence
        if item.get("relevance_gate", "ACCEPTED") == "ACCEPTED"
        and item.get("retrieval_score", 0.0) >= RELEVANCE_THRESHOLD
    ][:5]

    if final_verdict_text == "INSUFFICIENT_EVIDENCE":
        return (
            f'TruthLens could not establish the claim: "{original_claim}". '
            f'{len(usable_evidence)} relevant evidence item(s) were retrieved, '
            "but the system did not obtain a sufficiently consistent signal "
            "to support or contradict the claim. This is an inconclusive result, "
            "not a determination that the claim is true or false."
        )

    if not usable_evidence:
        evidence_text = "No sufficiently relevant evidence was retrieved."
    else:
        blocks = []
        for i, item in enumerate(usable_evidence, 1):
            blocks.append(
                f"Evidence {i}\nSource: {item.get('website', 'unknown source')}\n"
                f"Title: {item.get('title', '')}\n"
                f"Text: {str(item.get('evidence', ''))[:2500]}"
            )
        evidence_text = "\n\n".join(blocks)

    prompt = f"""You are explaining a verdict produced by TruthLens.
Original claim: {original_claim}
Authoritative verdict: {final_verdict_text}
Confidence: {format_confidence(final_confidence)}
Retrieved evidence:
{evidence_text}

Explain the verdict briefly using only the evidence above. Do not change,
contradict, or overstate the authoritative verdict. Do not add facts that
are not in the evidence. If the evidence does not directly establish the
claim, say so clearly."""
    response = ollama(prompt, timeout=180).strip()
    if response:
        return response
    return (
        f'The current evidence leads TruthLens to the verdict '
        f'{final_verdict_text} for "{original_claim}". '
        f'Confidence: {format_confidence(final_confidence)}.'
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    original_post,
    extracted_claim,
    atomic_claims,
    atomic_results,
    final_verdict_text,
    final_confidence,
    explanation
):

    result_row = {

        "original_post":
            original_post,

        "extracted_claim":
            extracted_claim,

        "atomic_claims":
            " | ".join(
                atomic_claims
            ),

        "final_verdict":
            final_verdict_text,

        "confidence":
            final_confidence,

        "explanation":
            explanation
    }

    pd.DataFrame(
        [result_row]
    ).to_csv(
        RESULT_FILE,
        index=False
    )

    detail_rows = []

    for claim, result in zip(
        atomic_claims,
        atomic_results
    ):

        for item in result[
            "evidence"
        ]:

            detail_rows.append({

                "atomic_claim":
                    claim,

                "atomic_verdict":
                    result["verdict"],

                "atomic_confidence":
                    result["confidence"],

                "evidence_id":
                    item.get(
                        "evidence_id",
                        ""
                    ),

                "source":
                    item.get(
                        "source",
                        ""
                    ),

                "website":
                    item.get(
                        "website",
                        ""
                    ),

                "title":
                    item.get(
                        "title",
                        ""
                    ),

                "retrieval_score":
                    item.get(
                        "retrieval_score",
                        0.0
                    ),

                "nli_label":
                    item.get(
                        "nli_label",
                        ""
                    ),

                "nli_confidence":
                    item.get(
                        "nli_confidence",
                        0.0
                    ),

                "temporal_score":
                    item.get(
                        "temporal_score",
                        0.0
                    ),

                "credibility":
                    item.get(
                        "credibility",
                        0.0
                    ),

                "evidence_weight":
                    item.get(
                        "evidence_weight",
                        0.0
                    ),

                "relevance_gate":
                    item.get(
                        "relevance_gate",
                        ""
                    ),

                "evidence":
                    item.get(
                        "evidence",
                        ""
                    ),

                "nli_passage":
                    item.get(
                        "nli_passage",
                        ""
                    ),

                "publish_date":
                    item.get(
                        "publish_date",
                        ""
                    ),

                "url":
                    item.get(
                        "url",
                        ""
                    )
            })

    pd.DataFrame(
        detail_rows
    ).to_csv(
        DETAIL_FILE,
        index=False
    )


# ============================================================
# MAIN TRUTHLENS PIPELINE
# ============================================================

def run_truthlens(
    social_media_post
):

    print()
    print("=" * 70)
    print("TRUTHLENS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CLAIM EXTRACTION
    # --------------------------------------------------------

    extracted_claim = extract_claim(
        social_media_post
    )

    print()
    print("Extracted Claim:")
    print(extracted_claim)

    # --------------------------------------------------------
    # 2. CLAIM DECOMPOSITION
    # --------------------------------------------------------

    atomic_claims = decompose_claim(
        extracted_claim
    )

    print()
    print("Atomic Claims:")

    for i, claim in enumerate(
        atomic_claims,
        1
    ):

        print(
            f"{i}. {claim}"
        )

    # --------------------------------------------------------
    # 3. PROCESS EACH ATOMIC CLAIM
    # --------------------------------------------------------

    atomic_results = []

    all_evidence_for_rag = []

    for claim in atomic_claims:

        print()
        print(
            f"Processing: {claim}"
        )

        # ----------------------------------------------------
        # Detect current/date-sensitive claim
        # ----------------------------------------------------

        current_claim = (
            is_current_claim(claim)
        )

        if current_claim:

            print(
                "  Current/date-sensitive "
                "claim detected."
            )

        # ----------------------------------------------------
        # Historical retrieval
        # ----------------------------------------------------

        historical_results = (
            historical_retrieval(
                claim
            )
        )

        # ----------------------------------------------------
        # Real-time retrieval
        # ----------------------------------------------------

        realtime_results = (
            realtime_retrieval(
                claim
            )
        )

        # ----------------------------------------------------
        # Evidence ranking
        # ----------------------------------------------------

        ranked_evidence = rank_evidence(
            claim,
            historical_results,
            realtime_results
        )

        # ----------------------------------------------------
        # Resolve claim
        # ----------------------------------------------------

        result = resolve_atomic_claim(
            claim,
            ranked_evidence,
            current_claim=current_claim
        )

        atomic_results.append(
            result
        )

        all_evidence_for_rag.extend(
            result["evidence"]
        )

        # ----------------------------------------------------
        # Display evidence
        # ----------------------------------------------------

        for item in result[
            "evidence"
        ]:

            print(
                f"  "
                f"{item.get('nli_label', 'NEUTRAL').upper():15}"
                f" "
                f"{item.get('nli_confidence', 0.0):.3f}"
                f" | "
                f"{item.get('source', ''):11}"
                f" | Retrieval: "
                f"{item.get('retrieval_score', 0.0):.3f}"
                f" | Temporal: "
                f"{item.get('temporal_score', 1.0):.3f}"
                f" | Credibility: "
                f"{item.get('credibility', 1.0):.2f}"
                f" | Gate: "
                f"{item.get('relevance_gate', '')}"
                f"{' | ROLE-REVERSED' if item.get('role_reversal') else ''}"
            )

        print(
            f"  → Atomic Verdict: "
            f"{result['verdict']} "
            f"| Confidence: "
            f"{format_confidence(result['confidence'])}"
        )

    # --------------------------------------------------------
    # 4. FINAL VERDICT
    # --------------------------------------------------------

    final_verdict_text, final_confidence = (
        final_verdict(
            atomic_results
        )
    )

    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print()
    print(
        f"Verdict: "
        f"{final_verdict_text}"
    )

    print(
        f"Confidence: "
        f"{format_confidence(final_confidence)}"
    )

    print()
    print("Atomic Verdict Summary:")

    for claim, result in zip(
        atomic_claims,
        atomic_results
    ):

        print(
            f"- {claim}"
        )

        print(
            f"  {result['verdict']} "
            f"({format_confidence(result['confidence'])})"
        )

    # --------------------------------------------------------
    # 5. RAG EXPLANATION
    # --------------------------------------------------------

    print()
    print(
        "Generating RAG explanation..."
    )

    explanation = generate_explanation(
        extracted_claim,
        final_verdict_text,
        final_confidence,
        all_evidence_for_rag
    )

    print()
    print("Explanation:")
    print(explanation)

    # --------------------------------------------------------
    # 6. SAVE
    # --------------------------------------------------------

    save_results(
        social_media_post,
        extracted_claim,
        atomic_claims,
        atomic_results,
        final_verdict_text,
        final_confidence,
        explanation
    )

    print()
    print("Saved:")
    print(RESULT_FILE)

    print("Evidence details:")
    print(DETAIL_FILE)

    return {
        "claim":
            extracted_claim,

        "atomic_claims":
            atomic_claims,

        "verdict":
            final_verdict_text,

        "confidence":
            final_confidence,

        "explanation":
            explanation
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "TRUTHLENS - EVIDENCE-AWARE "
        "MISINFORMATION VERIFICATION"
    )
    print("=" * 70)

    print()

    social_media_post = input(
        "Enter social-media post:\n> "
    ).strip()

    if not social_media_post:

        print(
            "No input provided."
        )

    else:

        run_truthlens(
            social_media_post
        )