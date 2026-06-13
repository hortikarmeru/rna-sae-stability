"""Seed stability test: how many SAE features reappear when only the random seed changes.

Usage: python scripts/04_stability.py l1

Compares the first two seeds in sae.seeds on the first layer in model.layers: features of the two
SAEs are matched one-to-one by decoder cosine similarity (Hungarian algorithm), and a matched
feature counts as stable if its similarity is >= analysis.cosine_threshold.
"""
import sys

import torch

from rna_sae.config import load_config
from rna_sae.features import match_features, unit_decoder
from rna_sae.sae import ARCHES, load_sae


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 04_stability.py {{{','.join(ARCHES)}}}")
    config = load_config()
    threshold = config["analysis"]["cosine_threshold"]
    layer = config["model"]["layers"][0]
    seed_a, seed_b = config["sae"]["seeds"][:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_a, _, _ = load_sae(config, arch, layer, seed_a, device)
    sae_b, _, _ = load_sae(config, arch, layer, seed_b, device)
    _, _, sims = match_features(unit_decoder(sae_a), unit_decoder(sae_b))
    n_stable = int((sims >= threshold).sum())
    print(f"Layer {layer}, seeds {seed_a} vs {seed_b}: {n_stable}/{len(sims)} matched features stable "
          f"({100 * n_stable / len(sims):.1f}%), mean matched similarity {sims.mean().item():.4f}")


if __name__ == "__main__":
    main()
