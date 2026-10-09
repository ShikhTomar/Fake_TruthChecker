import runpy
from pathlib import Path

# Reuse the existing rankings and test different RRF constants.
# The original script will be executed separately for each setting.

source = Path(__file__).with_name("hybrid_retrieval.py")
text = source.read_text(encoding="utf-8")

results = []

for rrf_k in [10, 30, 60, 100]:
    for top_k in [1, 3, 5]:
        modified = text.replace("RRF_K = 60", f"RRF_K = {rrf_k}")
        modified = modified.replace("TOP_K = 5", f"TOP_K = {top_k}")

        # Save a temporary variant beside the original script.
        temp = source.with_name("_hybrid_temp.py")
        temp.write_text(modified, encoding="utf-8")

        print(f"\nTesting RRF_K={rrf_k}, TOP_K={top_k}")
        runpy.run_path(str(temp), run_name="__main__")

        import pandas as pd
        output = Path.home() / "Downloads" / "TruthLens" / \
            "hybrid_retrieval_results.csv"
        df = pd.read_csv(output)

        n = len(df)
        ranks = df["gold_rank"]

        results.append({
            "RRF_K": rrf_k,
            "TOP_K": top_k,
            "Recall@1": ((ranks > 0) & (ranks <= 1)).sum() / n,
            "Recall@3": ((ranks > 0) & (ranks <= 3)).sum() / n,
            "Recall@5": ((ranks > 0) & (ranks <= 5)).sum() / n,
            "MRR": sum(1 / r if r > 0 else 0 for r in ranks) / n
        })

temp = source.with_name("_hybrid_temp.py")
if temp.exists():
    temp.unlink()

summary = pd.DataFrame(results)
out = Path.home() / "Downloads" / "TruthLens" / \
    "hyperparameter_tuning_results.csv"
summary.to_csv(out, index=False)

print("\nTUNING SUMMARY")
print(summary.to_string(index=False))
print(f"\nSaved to: {out}")