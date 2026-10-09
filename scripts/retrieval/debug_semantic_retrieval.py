import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path.home() / "Downloads" / "TruthLens"

INDEX_FILE = BASE / "semantic_index" / "evidence.faiss"
MAPPING_FILE = BASE / "semantic_index" / "evidence_mapping.csv"
BM25_FILE = BASE / "bm25_test_results.csv"
GOLD_FILE = BASE / "bm25_evaluation_mapping.csv"

# Load
index = faiss.read_index(str(INDEX_FILE))
evidence = pd.read_csv(MAPPING_FILE, encoding="utf-8-sig")
bm25 = pd.read_csv(BM25_FILE, encoding="utf-8-sig")
gold = pd.read_csv(GOLD_FILE, encoding="utf-8-sig")

print("FAISS vectors:", index.ntotal)
print("Mapping rows:", len(evidence))

print("\nFirst evidence IDs:")
print(evidence["evidence_id"].head(10).tolist())

# Take first 5 unique claims
claims = bm25[["claim_id", "claim"]].drop_duplicates("claim_id").head(5)

claims = claims.merge(
    gold[["id", "gold_evidence_id"]],
    left_on="claim_id",
    right_on="id",
    how="left"
)

model = SentenceTransformer("intfloat/multilingual-e5-small")

queries = [
    "query: " + str(x)
    for x in claims["claim"]
]

embeddings = model.encode(
    queries,
    batch_size=5,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(embeddings, 5)

evidence_ids = evidence["evidence_id"].astype(str).tolist()

print("\n" + "=" * 70)
print("DEBUG RESULTS")
print("=" * 70)

for i in range(len(claims)):

    gold_id = str(claims.iloc[i]["gold_evidence_id"])

    retrieved = [
        evidence_ids[idx]
        for idx in indices[i]
    ]

    print("\nClaim ID:", claims.iloc[i]["claim_id"])
    print("Gold evidence:", gold_id)
    print("Retrieved:", retrieved)
    print("Gold found:", gold_id in retrieved)

    print("\nClaim:")
    print(str(claims.iloc[i]["claim"])[:300])

print("\nDebug complete.")