"""
analyze_feature_biology_topk.py

Same as analyze_feature_biology_v2.py, adapted for the TopK SAE
checkpoints. For the layer-9, seed-0 TopK SAE: computes each feature's
average stability (cosine similarity matched against all other 9 seeds),
then runs the bpRNA-90 labeled data through it and computes structural
enrichment per feature. Joins the two and tests the core hypothesis:
do stable features show stronger structural enrichment than unstable ones?

Usage:
    python analyze_feature_biology_topk.py
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from collections import Counter
import csv

# ---- Config -------------------------------------------------------------
LAYER = 9
SEED = 0
ALL_SEEDS = list(range(10))
CHECKPOINT_DIR = Path("sae_checkpoints_topk")
BPRNA_PATH = Path("bprna_activations") / f"bprna_layer_{LAYER}.pt"
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"
OUTPUT_DIR = Path("biology_results_topk")
EVAL_BATCH_SIZE = 4096
MIN_FIRINGS_TO_REPORT = 30
COSINE_THRESHOLD = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = {
    "S": "Stem", "H": "Hairpin Loop", "I": "Internal Loop",
    "B": "Bulge", "M": "Multiloop", "E": "External Loop", "X": "Ambiguous",
}


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

    def forward(self, x):
        pre_activations = self.encoder(x)
        topk_values, topk_indices = torch.topk(pre_activations, self.k, dim=1)
        topk_values = torch.relu(topk_values)
        features = torch.zeros_like(pre_activations)
        features.scatter_(1, topk_indices, topk_values)
        reconstruction = self.decoder(features)
        return reconstruction, features


def load_sae(layer, seed):
    path = CHECKPOINT_DIR / f"sae_topk_layer{layer}_seed{seed}.pt"
    checkpoint = torch.load(path, weights_only=True)
    config = checkpoint["config"]
    sae = TopKSparseAutoencoder(config["input_dim"], config["dict_size"], config["k"]).to(DEVICE)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    mean = checkpoint["mean"].to(DEVICE)
    std = checkpoint["std"].to(DEVICE)
    return sae, mean, std, config


def find_alive_features(sae, normalized_vectors_cpu, dict_size):
    ever_active = torch.zeros(dict_size, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        for i in range(0, normalized_vectors_cpu.shape[0], EVAL_BATCH_SIZE):
            batch = normalized_vectors_cpu[i:i + EVAL_BATCH_SIZE].to(DEVICE)
            _, features = sae(batch)
            ever_active |= (features > 0).any(dim=0)
    return ever_active


def get_normalized_decoder(sae, alive_mask):
    with torch.no_grad():
        decoder_weight = sae.decoder.weight
        alive_weight = decoder_weight[:, alive_mask]
        norms = alive_weight.norm(dim=0, keepdim=True)
        normalized = alive_weight / (norms + 1e-8)
    return normalized.detach()


print(f"Loading activations from {ACTIVATIONS_PATH} (for alive-feature detection)...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)
all_vectors_cpu = torch.cat(sequences, dim=0).float()
del sequences

print(f"Preparing target SAE: layer {LAYER}, seed {SEED}...")
sae_target, mean_target, std_target, config_target = load_sae(LAYER, SEED)
dict_size = config_target["dict_size"]
normalized_target_cpu = (all_vectors_cpu - mean_target.cpu()) / (std_target.cpu() + 1e-6)
alive_target = find_alive_features(sae_target, normalized_target_cpu, dict_size)
decoder_target = get_normalized_decoder(sae_target, alive_target)

alive_target_indices = torch.nonzero(alive_target).squeeze(-1).tolist()
print(f"Target SAE alive features: {len(alive_target_indices)}")

per_feature_similarities = {idx: [] for idx in alive_target_indices}

for other_seed in ALL_SEEDS:
    if other_seed == SEED:
        continue
    print(f"  Comparing against seed {other_seed}...")
    sae_other, mean_other, std_other, config_other = load_sae(LAYER, other_seed)
    normalized_other_cpu = (all_vectors_cpu - mean_other.cpu()) / (std_other.cpu() + 1e-6)
    alive_other = find_alive_features(sae_other, normalized_other_cpu, dict_size)
    decoder_other = get_normalized_decoder(sae_other, alive_other)

    with torch.no_grad():
        similarity_matrix = decoder_target.T @ decoder_other
        cost_matrix = (-similarity_matrix).cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_sims = similarity_matrix[row_indices, col_indices].cpu().tolist()

    for local_row, sim in zip(row_indices, matched_sims):
        global_feat_idx = alive_target_indices[local_row]
        per_feature_similarities[global_feat_idx].append(sim)

    del sae_other, normalized_other_cpu
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

avg_stability = {
    idx: sum(sims) / len(sims) if sims else 0.0
    for idx, sims in per_feature_similarities.items()
}

del all_vectors_cpu, normalized_target_cpu

print(f"\nLoading {BPRNA_PATH}...")
bprna_data = torch.load(BPRNA_PATH, weights_only=False)
vectors = bprna_data["vectors"].float()
structural_labels = bprna_data["structural_labels"]
print(f"Total labeled nucleotide vectors: {vectors.shape[0]}")

normalized_bprna_cpu = (vectors - mean_target.cpu()) / (std_target.cpu() + 1e-6)

overall_counts = Counter(structural_labels)
total_n = len(structural_labels)
baseline_rate = {label: count / total_n for label, count in overall_counts.items()}

print(f"\nOverall dataset label distribution (baseline):")
for label, rate in sorted(baseline_rate.items(), key=lambda x: -x[1]):
    print(f"  {label} ({LABEL_NAMES.get(label, '?')}): {100*rate:.1f}%")

feature_label_counts = {i: Counter() for i in range(dict_size)}
feature_total_firings = [0] * dict_size

print(f"\nRunning {vectors.shape[0]} vectors through SAE...")
with torch.no_grad():
    for i in range(0, normalized_bprna_cpu.shape[0], EVAL_BATCH_SIZE):
        batch = normalized_bprna_cpu[i:i + EVAL_BATCH_SIZE].to(DEVICE)
        batch_labels = structural_labels[i:i + EVAL_BATCH_SIZE]

        _, features = sae_target(batch)
        fired = (features > 0).cpu()

        for row_idx in range(fired.shape[0]):
            label = batch_labels[row_idx]
            fired_feature_indices = torch.nonzero(fired[row_idx]).squeeze(-1).tolist()
            if isinstance(fired_feature_indices, int):
                fired_feature_indices = [fired_feature_indices]
            for feat_idx in fired_feature_indices:
                feature_label_counts[feat_idx][label] += 1
                feature_total_firings[feat_idx] += 1

print("  done")

OUTPUT_DIR.mkdir(exist_ok=True)
results = []

for feat_idx in range(dict_size):
    total_firings = feature_total_firings[feat_idx]
    if total_firings < MIN_FIRINGS_TO_REPORT:
        continue

    counts = feature_label_counts[feat_idx]
    best_label, best_enrichment = None, 0.0
    for label, count in counts.items():
        feature_rate = count / total_firings
        enrichment = feature_rate / baseline_rate.get(label, 1e-6)
        if enrichment > best_enrichment:
            best_label, best_enrichment = label, enrichment

    stability = avg_stability.get(feat_idx, None)

    results.append({
        "feature_idx": feat_idx,
        "total_firings": total_firings,
        "best_label": best_label,
        "best_label_name": LABEL_NAMES.get(best_label, "?"),
        "max_enrichment": round(best_enrichment, 2),
        "avg_stability": round(stability, 3) if stability is not None else None,
        "is_stable": stability is not None and stability >= COSINE_THRESHOLD,
    })

results.sort(key=lambda r: -r["max_enrichment"])

print(f"\n{'='*70}")
print(f"Found {len(results)} features with >= {MIN_FIRINGS_TO_REPORT} firings")
print(f"{'='*70}")

print(f"\nTop 15 most ENRICHED features (firing rate vs baseline):")
for r in results[:15]:
    print(f"  Feature {r['feature_idx']:5d} | fires {r['total_firings']:5d}x | "
          f"label: {r['best_label']} ({r['best_label_name']}) | "
          f"enrichment: {r['max_enrichment']}x | "
          f"stability: {r['avg_stability']} | stable: {r['is_stable']}")

print(f"\nBottom 10 LEAST enriched features:")
for r in results[-10:]:
    print(f"  Feature {r['feature_idx']:5d} | fires {r['total_firings']:5d}x | "
          f"label: {r['best_label']} ({r['best_label_name']}) | "
          f"enrichment: {r['max_enrichment']}x | "
          f"stability: {r['avg_stability']} | stable: {r['is_stable']}")

stable_enrichments = [r["max_enrichment"] for r in results if r["is_stable"]]
unstable_enrichments = [r["max_enrichment"] for r in results if not r["is_stable"]]

print(f"\n{'='*70}")
print("KEY RESULT: does stability correlate with structural enrichment?")
print(f"{'='*70}")
if stable_enrichments:
    print(f"Stable features (n={len(stable_enrichments)}): "
          f"avg enrichment = {sum(stable_enrichments)/len(stable_enrichments):.2f}x")
else:
    print("No stable features found among reportable features.")
if unstable_enrichments:
    print(f"Unstable features (n={len(unstable_enrichments)}): "
          f"avg enrichment = {sum(unstable_enrichments)/len(unstable_enrichments):.2f}x")

csv_path = OUTPUT_DIR / f"feature_enrichment_stability_topk_layer{LAYER}_seed{SEED}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved full results to {csv_path}")
