"""FastAPI wrapper for the existing TruthLens verification pipeline.

Run from the TruthLens project folder with:
    uvicorn truthlens_api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importing this module loads TruthLens' existing corpus, FAISS index, and models.
# Keep truthlens.py and its data/index files in the same project setup as before.
import truthlens

app = FastAPI(
    title="TruthLens API",
    version="1.0.0",
    description="API wrapper around the existing TruthLens claim-verification pipeline.",
)

# Vite's default development origin. Restrict this to your frontend origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# The existing pipeline writes shared CSV output files. Serialize verification requests
# so simultaneous requests cannot race while writing those files or using shared models.
_verification_lock = threading.Lock()


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000, description="Social-media post or claim to verify")

    @classmethod
    def validate_text(cls, value: str) -> str:
        return value.strip()


class VerifyResponse(BaseModel):
    success: bool
    claim: str
    atomic_claims: list[str]
    verdict: str
    confidence: float | None
    explanation: str


@app.get("/health")
def health() -> dict[str, str]:
    """Simple liveness endpoint; it does not re-run model loading."""
    return {"status": "ok", "service": "TruthLens API"}


@app.post("/verify", response_model=VerifyResponse)
def verify_claim(request: VerifyRequest) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Please provide a non-empty claim or social-media post.")

    try:
        with _verification_lock:
            result = truthlens.run_truthlens(text)

        # Normalize output for JSON. Do not invent evidence fields the current pipeline
        # does not return yet; evidence cards can be added in a later integration step.
        confidence = result.get("confidence")
        return {
            "success": True,
            "claim": str(result.get("claim", "")),
            "atomic_claims": [str(x) for x in result.get("atomic_claims", [])],
            "verdict": str(result.get("verdict", "INSUFFICIENT_EVIDENCE")),
            "confidence": float(confidence) if confidence is not None else None,
            "explanation": str(result.get("explanation", "")),
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Keep internal stack traces out of the API response; details remain in server logs.
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Verification failed: {type(exc).__name__}. Check the backend terminal.") from exc
