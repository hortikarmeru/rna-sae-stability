"""Seed stability: how many SAE features reappear when only the random seed changes.

Usage: python scripts/04_stability.py {l1,topk}

For each layer, every pair of seeds (45 pairs for 10 seeds) is compared: alive features of the
two SAEs are matched one-to-one by decoder cosine similarity (Hungarian algorithm), and a
matched feature counts as stable if its similarity is >= analysis.cosine_threshold.
Writes results/<arch>/stability_pairs.csv (one row per pair) and
results/<arch>/stability_summary.csv (per layer: mean % stable and mean similarity).
"""
import csv
import sys
from itertools import combinations

import numpy as np
import torch

from rna_sae.activations import load_rnacentral_vectors, normalize
from rna_sae.config import load_config, repo_path
from rna_sae.features import firing_counts, match_features, unit_decoder
from rna_sae.meta import write_meta
from rna_sae.sae import ARCHES, load_sae

def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 04_stability.py {{{','.join(ARCHES)}}}")
    config = load_config()
    eps, batch = config["sae"]["normalize_eps"], config["analysis"]["eval_batch_size"]
    threshold = config["analysis"]["cosine_threshold"]
    seeds = config["sae"]["seeds"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pair_rows, summary_rows = [], []
    for layer in config["model"]["layers"]:
        vectors = load_rnacentral_vectors(config, layer)
        decoders, n_alive = {}, {}
        for seed in seeds:
            sae, mean, std = load_sae(config, arch, layer, seed, device)
            alive = firing_counts(sae, normalize(vectors, mean, std, eps), batch, device) > 0
            decoders[seed], n_alive[seed] = unit_decoder(sae, alive), int(alive.sum())
            del sae
        del vectors

        for a, b in combinations(seeds, 2):
            _, _, sims = match_features(decoders[a], decoders[b])
            n_stable = int((sims >= threshold).sum())
            pair_rows.append({
                "layer": layer, "seed_a": a, "seed_b": b, "n_alive_a": n_alive[a], "n_alive_b": n_alive[b],
                "n_matched": len(sims), "n_stable": n_stable,
                "pct_stable": round(100 * n_stable / len(sims), 2) if len(sims) else 0.0,
                "mean_matched_similarity": round(sims.mean().item(), 4) if len(sims) else 0.0,
            })
        layer_rows = [r for r in pair_rows if r["layer"] == layer]
        summary_rows.append({
            "layer": layer, "n_pairs": len(layer_rows),
            "mean_n_alive": round(float(np.mean(list(n_alive.values()))), 1),
            "mean_pct_stable": round(float(np.mean([r["pct_stable"] for r in layer_rows])), 2),
            "mean_matched_similarity": round(float(np.mean([r["mean_matched_similarity"] for r in layer_rows])), 4),
        })
        print(summary_rows[-1], flush=True)

    out_dir = repo_path(config["paths"]["results_dir"]) / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stability_pairs.csv", pair_rows), ("stability_summary.csv", summary_rows)):
        with open(out_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        write_meta(out_dir / name, config, packages=["torch", "scipy", "numpy"])


if __name__ == "__main__":
    main()
