"""Train one SAE per layer x seed on the cached RNAcentral activations.

Usage: python scripts/02_train_saes.py {l1,topk}

Activations are z-scored per dimension (mean and std saved in each checkpoint), then an SAE is
trained for every layer in model.layers and every seed in sae.seeds with identical settings.
Checkpoints go to paths.checkpoints; a per-epoch loss log goes to results/<arch>/training.json.
"""
import json
import sys

import torch

from rna_sae.activations import load_rnacentral_vectors, normalize
from rna_sae.config import data_path, load_config, repo_path
from rna_sae.meta import write_meta
from rna_sae.sae import ARCHES, save_sae, train_sae


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 02_train_saes.py {{{','.join(ARCHES)}}}")
    config = load_config()
    sae_cfg = config["sae"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log = {}
    for layer in config["model"]["layers"]:
        vectors = load_rnacentral_vectors(config, layer)
        mean, std = vectors.mean(dim=0, keepdim=True), vectors.std(dim=0, keepdim=True)
        normalized = normalize(vectors, mean, std, sae_cfg["normalize_eps"])
        del vectors
        print(f"Layer {layer}: {normalized.shape[0]} token vectors", flush=True)
        for seed in sae_cfg["seeds"]:
            sae, history = train_sae(arch, sae_cfg, normalized, seed, device)
            save_sae(data_path(config, "checkpoints", arch=arch, layer=layer, seed=seed),
                     sae, arch, sae_cfg, layer, seed, mean, std, history)
            log[f"layer{layer}_seed{seed}"] = history
        del normalized

    out = repo_path(config["paths"]["results_dir"]) / arch / "training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(log, f, indent=1)
    write_meta(out, config, packages=["torch"])


if __name__ == "__main__":
    main()
