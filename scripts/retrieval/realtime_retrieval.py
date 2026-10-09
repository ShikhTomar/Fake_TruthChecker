import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

BASE = Path.home() / "Downloads" / "TruthLens"

OUTPUT_FILE = BASE / "realtime_evidence.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/154.0 Safari/537.36"
    )
}


# ============================================================
# EXTRACT ARTICLE TEXT
# ============================================================

def extract_article(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove irrelevant elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form"
        ]):
            tag.decompose()

        # Prefer article tag
        article = soup.find("article")

        if article:
            text = article.get_text(
                " ",
                strip=True
            )
        else:
            paragraphs = soup.find_all("p")

            text = " ".join(
                p.get_text(
                    " ",
                    strip=True
                )
                for p in paragraphs
            )

        # Normalize whitespace
        text = " ".join(text.split())

        return text

    except Exception:
        return ""


# ============================================================
# CLAIM
# ============================================================

claim = input(
    "\nEnter claim to fact-check: "
).strip()

if not claim:
    raise ValueError(
        "Claim cannot be empty."
    )


# ============================================================
# SEARCH
# ============================================================

print("\nSearching the web...")

with DDGS() as ddgs:

    search_results = list(
        ddgs.text(
            claim,
            max_results=10
        )
    )


if not search_results:

    print("No web results found.")
    exit()


# ============================================================
# FETCH ARTICLES
# ============================================================

rows = []

print(
    f"\nFound {len(search_results)} search results."
)

print(
    "Fetching article content..."
)

for rank, result in enumerate(
    search_results,
    start=1
):

    title = result.get(
        "title",
        ""
    )

    url = result.get(
        "href",
        ""
    )

    snippet = result.get(
        "body",
        ""
    )

    print(
        f"\n[{rank}/{len(search_results)}] "
        f"{title}"
    )

    article_text = extract_article(
        url
    )

    if article_text:

        print(
            "Article extracted:",
            len(article_text),
            "characters"
        )

    else:

        print(
            "Article extraction failed — "
            "keeping search snippet."
        )

        article_text = snippet

    domain = urlparse(url).netloc

    rows.append({

        "claim": claim,

        "retrieval_rank": rank,

        "title": title,

        "website": domain,

        "url": url,

        "snippet": snippet,

        "evidence": article_text,

        "retrieved_at":
            datetime.now().isoformat()

    })


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "REAL-TIME EVIDENCE EXTRACTION COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nResults:",
    len(df)
)

print(
    "Articles with extracted content:",
    sum(
        df["evidence"].str.len() > 0
    )
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)