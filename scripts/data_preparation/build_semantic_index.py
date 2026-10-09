import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path.home() / "Downloads" / "TruthLens"
DATA = BASE / "factdrill_evidence_corpus.csv"
OUT = BASE / "semantic_index"
OUT.mkdir(exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"

# Load evidence
df = pd.read_csv(DATA).fillna("")
print("Evidence documents:", len(df))

# Load multilingual embedding model
model = SentenceTransformer(MODEL_NAME)

# E5 expects a passage prefix for evidence
#df = df.head(2000)
texts = [
    "passage: " + str(text)
    for text in df["investigation"]
]

# Generate embeddings in batches
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

# Create FAISS cosine-similarity index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and document mapping
faiss.write_index(
    index,
    str(OUT / "evidence.faiss")
)

df[["evidence_id", "title", "website", "investigation"]].to_csv(
    OUT / "evidence_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

print("FAISS vectors:", index.ntotal)
print("Index saved successfully!")