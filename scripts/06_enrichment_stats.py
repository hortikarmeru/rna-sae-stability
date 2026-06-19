"""Statistics for the stability-enrichment relationship (reads the CSV from 05).

Usage: python scripts/06_enrichment_stats.py {l1,topk}

For all reported features, and again for features with >= analysis.high_confidence_min_firings:
  - Pearson r between avg_stability and max_enrichment (raw and log)
  - Welch's t-test of mean enrichment, stable vs unstable features
  - partial correlation of stability and log enrichment controlling for log firing count.
    Rarely firing features are both less stable and more extreme in enrichment (a maximum over
    labels from few firings is noisy), so firing count can inflate the raw correlation.
Writes results/<arch>/enrichment_stats.json.
"""
import csv
import json
import sys

import numpy as np
from scipy import stats

from rna_sae.config import load_config, repo_path
from rna_sae.meta import write_meta
from rna_sae.sae import ARCHES


def residuals(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def analyze(rows):
    s = np.array([float(r["avg_stability"]) for r in rows])
    e = np.array([float(r["max_enrichment"]) for r in rows])
    log_e = np.log(np.where(e > 0, e, 1.0))
    log_n = np.log(np.array([int(r["total_firings"]) for r in rows]))
    stable = np.array([r["is_stable"] == "True" for r in rows])
    r_raw, p_raw = stats.pearsonr(s, e)
    r_log, p_log = stats.pearsonr(s, log_e)
    t, p_t = stats.ttest_ind(e[stable], e[~stable], equal_var=False)
    r_part, p_part = stats.pearsonr(residuals(s, log_n), residuals(log_e, log_n))
    return {
        "n": len(rows), "n_stable": int(stable.sum()), "n_unstable": int((~stable).sum()),
        "mean_enrichment_stable": round(float(e[stable].mean()), 3),
        "mean_enrichment_unstable": round(float(e[~stable].mean()), 3),
        "pearson_stability_enrichment": {"r": round(float(r_raw), 4), "p": float(p_raw)},
        "pearson_stability_log_enrichment": {"r": round(float(r_log), 4), "p": float(p_log)},
        "welch_t_stable_vs_unstable": {"t": round(float(t), 3), "p": float(p_t)},
        "partial_r_given_log_firings": {"r": round(float(r_part), 4), "p": float(p_part)},
        "r_stability_log_firings": round(float(stats.pearsonr(s, log_n)[0]), 4),
        "r_log_enrichment_log_firings": round(float(stats.pearsonr(log_e, log_n)[0]), 4),
    }


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 06_enrichment_stats.py {{{','.join(ARCHES)}}}")
    config = load_config()
    a = config["analysis"]
    results_dir = repo_path(config["paths"]["results_dir"]) / arch
    with open(results_dir / f"enrichment_layer{a['target_layer']}_seed{a['target_seed']}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["avg_stability"] not in ("", "None")]

    out_stats = {
        f"min_firings_{a['min_firings']}": analyze(rows),
        f"min_firings_{a['high_confidence_min_firings']}":
            analyze([r for r in rows if int(r["total_firings"]) >= a["high_confidence_min_firings"]]),
    }
    print(json.dumps(out_stats, indent=2))
    out = results_dir / "enrichment_stats.json"
    with open(out, "w") as f:
        json.dump(out_stats, f, indent=2)
    write_meta(out, config, packages=["scipy", "numpy"])


if __name__ == "__main__":
    main()
