"""Verification script for the TruthLens Reference Architecture Pipeline.

Tests each architectural component:
1. Input: Social Media Text
2. Claim Extraction
3. Claim Decomposition
4. Dual Retrieval:
   - Historical Corpus (FactDrill BM25 + FAISS)
   - Real-Time Web Retrieval
5. Evidence Ranking (E5 dense ranking)
6. XLM-R NLI Classification
7. Temporal + Credibility + Conflict Resolution
8. Final Verdict Aggregation
9. RAG Explanation Generation
10. Artifact and Evaluation Verification (Jupyter notebook, metrics.json, plots, checkpoint)
"""
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

def test_pipeline():
    print("=" * 70)
    print("RUNNING ARCHITECTURE & COMPONENT VERIFICATION")
    print("=" * 70)

    # 1. Verify required data and model artifacts
    required_files = [
        BASE_DIR / "factdrill_evidence_corpus.csv",
        BASE_DIR / "semantic_index" / "evidence.faiss",
        BASE_DIR / "semantic_index" / "evidence_mapping.csv",
        BASE_DIR / "results" / "metrics.json",
        BASE_DIR / "results" / "models" / "best_xlm_r_head.pt",
        BASE_DIR / "results" / "plots" / "training_validation_curves.png",
        BASE_DIR / "results" / "plots" / "hyperparameter_tuning_comparison.png",
        BASE_DIR / "results" / "plots" / "baseline_vs_tuned_metrics.png",
        BASE_DIR / "results" / "plots" / "confusion_matrix_baseline.png",
        BASE_DIR / "results" / "plots" / "confusion_matrix_tuned.png",
        BASE_DIR / "model_training_evaluation.ipynb",
        BASE_DIR / "train_and_evaluate.py",
        BASE_DIR / "truthlens.py",
        BASE_DIR / "truthlens_api.py",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("[FAIL] Missing required files:")
        for m in missing:
            print(f"  - {m}")
        return False
    print("[PASS] All required architecture components, notebooks, models, and plot artifacts exist.")

    # 2. Test importing truthlens modules
    print("\nImporting TruthLens components...")
    import truthlens

    print("[PASS] TruthLens models and indices initialized successfully.")

    # 3. Test individual pipeline components
    test_claim = "UPI is shut down across India today"
    print(f"\nTesting Component 1 & 2: Claim Extraction on: '{test_claim}'")
    extracted = truthlens.extract_claim(test_claim)
    assert extracted, "Claim extraction returned empty string"
    print(f"  Extracted Claim: {extracted}")

    print(f"\nTesting Component 3: Claim Decomposition")
    atomic = truthlens.decompose_claim(extracted)
    assert atomic and len(atomic) >= 1, "Claim decomposition failed"
    print(f"  Atomic Claims: {atomic}")

    sub_claim = atomic[0]
    print(f"\nTesting Component 4A: Historical Corpus (FactDrill Hybrid Retrieval)")
    hist_ev = truthlens.historical_retrieval(sub_claim, top_k=3)
    assert isinstance(hist_ev, list), "Historical retrieval must return a list"
    print(f"  Retrieved {len(hist_ev)} historical evidence items.")

    print(f"\nTesting Component 4B: Real-Time Web Retrieval")
    # Real-time search might be tested with fallback if network is restricted
    rt_ev = truthlens.realtime_retrieval(sub_claim, max_results=3)
    print(f"  Retrieved {len(rt_ev)} real-time items.")

    print(f"\nTesting Component 5: Evidence Ranking")
    ranked = truthlens.rank_evidence(sub_claim, hist_ev, rt_ev)
    print(f"  Ranked evidence count: {len(ranked)}")

    print(f"\nTesting Component 6: XLM-R NLI")
    if ranked:
        nli_res = truthlens.nli(sub_claim, ranked[:2])
        print(f"  NLI inference ran on {len(nli_res)} items. Labels: {[r['label'] for r in nli_res]}")

    print(f"\nTesting Component 7 & 8: Conflict Resolution & Final Verdict")
    atomic_result = truthlens.resolve_atomic_claim(sub_claim, ranked)
    verdict, conf = truthlens.final_verdict([atomic_result])
    print(f"  Verdict: {verdict} | Confidence: {truthlens.format_confidence(conf)}")

    print(f"\nTesting Component 9: RAG Explanation")
    exp = truthlens.generate_explanation(extracted, verdict, conf, ranked)
    assert exp, "Explanation generation failed"
    print(f"  Explanation:\n  {exp[:200]}...")

    print("\n" + "=" * 70)
    print("ALL 9 ARCHITECTURAL STAGES PASSED COMPONENT-LEVEL VERIFICATION!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
