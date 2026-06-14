"""Do stable features track RNA structure? Per-feature stability joined with structural enrichment.

Usage: python scripts/05_enrichment.py l1

For the SAE at analysis.target_layer / analysis.target_seed:
  stability   each alive feature's matched cosine similarity to its partner in every other seed
              (Hungarian matching, as in 04), averaged over the other seeds
  enrichment  on bpRNA-90 nucleotides: for each structure label, (share of the feature's firings
              with that label) / (share of all nucleotides with that label). A feature's
              max_enrichment is its highest ratio over labels; 1.0 means no preference.
Enrichment corrects for label frequency (about half of all nucleotides are stems), which raw
"purity" does not. Features with fewer than analysis.min_firings firings are left out.
Writes results/<arch>/enrichment_layer<L>_seed<S>.csv.
"""
import csv
import sys

import torch

from rna_sae.activations import load_rnacentral_vectors, normalize
from rna_sae.config import data_path, load_config, repo_path
from rna_sae.features import firing_counts, match_features, unit_decoder
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

    # Per-feature stability of the target SAE against every other seed.
    vectors = load_rnacentral_vectors(config, layer)
    sae, mean, std = load_sae(config, arch, layer, target, device)
    alive = firing_counts(sae, normalize(vectors, mean, std, eps), batch, device) > 0
    alive_idx = torch.nonzero(alive).squeeze(-1).tolist()
    target_decoder = unit_decoder(sae, alive)
    sims = {i: [] for i in alive_idx}
    for seed in config["sae"]["seeds"]:
        if seed == target:
            continue
        other, m, s = load_sae(config, arch, layer, seed, device)
        other_alive = firing_counts(other, normalize(vectors, m, s, eps), batch, device) > 0
        rows, _, matched = match_features(target_decoder, unit_decoder(other, other_alive))
        for r, sim in zip(rows, matched.tolist()):
            sims[alive_idx[r]].append(sim)
        del other
    del vectors
    stability = {i: sum(v) / len(v) for i, v in sims.items() if v}

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
        enrichment = (counts[:, feat] / totals[feat]) / baseline
        best = int(enrichment.argmax())
        stab = stability.get(feat)
        results.append({
            "feature_idx": feat, "total_firings": int(totals[feat]),
            "best_label": label_set[best], "best_label_name": label_names.get(label_set[best], "?"),
            "max_enrichment": round(enrichment[best].item(), 2),
            "avg_stability": round(stab, 3) if stab is not None else None,
        })
    results.sort(key=lambda r: -r["max_enrichment"])
    print(f"{len(results)} features with >= {a_cfg['min_firings']} firings")

    out = repo_path(config["paths"]["results_dir"]) / arch / f"enrichment_layer{layer}_seed{target}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    write_meta(out, config, packages=["torch", "scipy"])


if __name__ == "__main__":
    main()
