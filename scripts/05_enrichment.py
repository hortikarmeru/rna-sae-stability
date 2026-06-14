"""Which RNA structures do SAE features fire on? Per-feature label purity on bpRNA-90.

Usage: python scripts/05_enrichment.py l1

For the SAE at analysis.target_layer / analysis.target_seed, counts each feature's firings on
bpRNA-90 nucleotides by structure label. A feature's purity is the share of its firings that
fall on its most common label. Features with fewer than analysis.min_firings firings are left
out. Writes results/<arch>/purity_layer<L>_seed<S>.csv.
"""
import csv
import sys

import torch

from rna_sae.activations import normalize
from rna_sae.config import data_path, load_config, repo_path
from rna_sae.meta import write_meta
from rna_sae.sae import ARCHES, load_sae


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else ""
    if arch not in ARCHES:
        sys.exit(f"usage: 05_enrichment.py {{{','.join(ARCHES)}}}")
    config = load_config()
    a_cfg = config["analysis"]
    eps, batch = config["sae"]["normalize_eps"], a_cfg["eval_batch_size"]
    layer, target = a_cfg["target_layer"], a_cfg["target_seed"]
    label_names = config["data"]["bprna"]["labels"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, mean, std = load_sae(config, arch, layer, target, device)

    # Firing counts per structure label on bpRNA-90.
    bprna = torch.load(data_path(config, "bprna_activations", layer=layer), weights_only=False)
    labels = bprna["structural_labels"]
    label_set = sorted(set(labels))
    label_idx = torch.tensor([label_set.index(x) for x in labels])
    normalized = normalize(bprna["vectors"].float(), mean, std, eps)
    counts = torch.zeros(len(label_set), sae.decoder.weight.shape[1])
    with torch.no_grad():
        for i in range(0, normalized.shape[0], batch):
            fired = (sae(normalized[i:i + batch].to(device))[1] > 0).float().cpu()
            counts.index_add_(0, label_idx[i:i + batch], fired)
    baseline = torch.bincount(label_idx, minlength=len(label_set)).float() / len(labels)
    for name, rate in zip(label_set, baseline.tolist()):
        print(f"  baseline {label_names.get(name, name)}: {100 * rate:.1f}%")

    results = []
    totals = counts.sum(dim=0)
    for feat in torch.nonzero(totals >= a_cfg["min_firings"]).squeeze(-1).tolist():
        shares = counts[:, feat] / totals[feat]
        best = int(shares.argmax())
        results.append({
            "feature_idx": feat, "total_firings": int(totals[feat]),
            "best_label": label_set[best], "best_label_name": label_names.get(label_set[best], "?"),
            "purity": round(shares[best].item(), 3),
        })
    results.sort(key=lambda r: -r["purity"])
    print(f"{len(results)} features with >= {a_cfg['min_firings']} firings")

    out = repo_path(config["paths"]["results_dir"]) / arch / f"purity_layer{layer}_seed{target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    write_meta(out, config, packages=["torch"])


if __name__ == "__main__":
    main()
