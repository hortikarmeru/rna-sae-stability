"""Dead-feature and selectivity check for every trained SAE.

Usage: python scripts/03_feature_diagnostics.py {l1,topk}

For each layer x seed, counts how many dictionary features ever fire on the RNAcentral
activations (alive) and how selective the alive ones are: the share of inputs each fires on,
binned by analysis.selectivity_bins. A useful SAE has most features alive and almost all of
them selective. Writes results/<arch>/feature_diagnostics.csv.
"""
import csv
import sys

import torch

from rna_sae.activations import load_rnacentral_vectors, normalize
from rna_sae.config import load_config, repo_path
from rna_sae.features import firing_counts
from rna_sae.meta import write_meta
from rna_sae.sae import ARCHES, load_sae


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 03_feature_diagnostics.py {{{','.join(ARCHES)}}}")
    config = load_config()
    eps, batch = config["sae"]["normalize_eps"], config["analysis"]["eval_batch_size"]
    low, high = config["analysis"]["selectivity_bins"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for layer in config["model"]["layers"]:
        vectors = load_rnacentral_vectors(config, layer)
        for seed in config["sae"]["seeds"]:
            sae, mean, std = load_sae(config, arch, layer, seed, device)
            counts = firing_counts(sae, normalize(vectors, mean, std, eps), batch, device)
            rates = counts[counts > 0] / vectors.shape[0]
            n_alive = len(rates)
            rows.append({
                "layer": layer, "seed": seed, "dict_size": counts.numel(), "n_alive": n_alive,
                "pct_alive": round(100 * n_alive / counts.numel(), 2),
                "median_firing_rate": round(rates.median().item(), 6) if n_alive else None,
                "max_firing_rate": round(rates.max().item(), 6) if n_alive else None,
                "n_selective": int((rates < low).sum()),
                "n_moderate": int(((rates >= low) & (rates < high)).sum()),
                "n_universal": int((rates >= high).sum()),
            })
            print(rows[-1], flush=True)
        del vectors

    out = repo_path(config["paths"]["results_dir"]) / arch / "feature_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_meta(out, config, packages=["torch"])


if __name__ == "__main__":
    main()
