"""
analyze_feature_biology_topk_rigorous.py

Reuses the already-saved per-feature enrichment + stability CSV from
analyze_feature_biology_topk.py and adds real statistical rigor:
  1. Pearson correlation between avg_stability and max_enrichment
     (and log(enrichment), since enrichment ratios are naturally skewed)
  2. A t-test comparing stable vs unstable groups (is the difference in
     means plausibly due to chance, or real?)
  3. The same comparison restricted to higher-confidence features only
     (>=100 firings instead of >=30), as a robustness check against the
     small-sample noise in the original top-enrichment list.

This does NOT require re-running the SAE -- it reads the CSV already
saved by analyze_feature_biology_topk.py.

Usage:
    python analyze_feature_biology_topk_rigorous.py
"""

import csv
import math
from pathlib import Path
from scipy import stats

CSV_PATH = Path("biology_results_topk/feature_enrichment_stability_topk_layer9_seed0.csv")
HIGH_CONFIDENCE_MIN_FIRINGS = 100

print(f"Loading {CSV_PATH}...")
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "feature_idx": int(row["feature_idx"]),
            "total_firings": int(row["total_firings"]),
            "max_enrichment": float(row["max_enrichment"]),
            "avg_stability": float(row["avg_stability"]),
            "is_stable": row["is_stable"] == "True",
        })
print(f"Loaded {len(rows)} features")


def run_analysis(data, label):
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {label} (n={len(data)})")
    print(f"{'='*70}")

    if len(data) < 3:
        print("Too few features to analyze meaningfully.")
        return

    stabilities = [r["avg_stability"] for r in data]
    enrichments = [r["max_enrichment"] for r in data]
    log_enrichments = [math.log(e) if e > 0 else 0.0 for e in enrichments]

    pearson_r, pearson_p = stats.pearsonr(stabilities, enrichments)
    pearson_r_log, pearson_p_log = stats.pearsonr(stabilities, log_enrichments)

    print(f"\nPearson correlation (stability vs raw enrichment):")
    print(f"  r = {pearson_r:.3f}, p = {pearson_p:.4g}")
    print(f"Pearson correlation (stability vs log(enrichment)):")
    print(f"  r = {pearson_r_log:.3f}, p = {pearson_p_log:.4g}")

    stable_group = [r["max_enrichment"] for r in data if r["is_stable"]]
    unstable_group = [r["max_enrichment"] for r in data if not r["is_stable"]]

    if len(stable_group) >= 2 and len(unstable_group) >= 2:
        mean_stable = sum(stable_group) / len(stable_group)
        mean_unstable = sum(unstable_group) / len(unstable_group)

        t_stat, t_p = stats.ttest_ind(stable_group, unstable_group, equal_var=False)

        print(f"\nStable features (n={len(stable_group)}): mean enrichment = {mean_stable:.2f}x")
        print(f"Unstable features (n={len(unstable_group)}): mean enrichment = {mean_unstable:.2f}x")
        print(f"Difference: {mean_stable - mean_unstable:.2f}x")
        print(f"Welch's t-test: t = {t_stat:.3f}, p = {t_p:.4g}")
        if t_p < 0.05:
            print("  -> Statistically significant at p < 0.05")
        else:
            print("  -> NOT statistically significant at p < 0.05")
    else:
        print("\nNot enough features in one group to run a t-test.")


run_analysis(rows, "All features (>=30 firings, original threshold)")

high_confidence = [r for r in rows if r["total_firings"] >= HIGH_CONFIDENCE_MIN_FIRINGS]
run_analysis(high_confidence, f"High-confidence features only (>={HIGH_CONFIDENCE_MIN_FIRINGS} firings)")
