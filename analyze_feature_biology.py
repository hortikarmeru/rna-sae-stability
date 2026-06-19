"""
analyze_feature_biology.py

For one trained SAE, runs the labeled bpRNA-90 activations through it and,
for each alive feature, tabulates which structural categories (Stem,
Hairpin, Internal loop, Bulge, Multiloop, External loop, Ambiguous) it
fires on. Computes a "purity" score per feature (% of firings landing in
its single most common category) as a simple interpretability proxy.

Usage:
    python analyze_feature_biology.py
"""

import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
import csv

# ---- Config -------------------------------------------------------------
LAYER = 9
SEED = 0
CHECKPOINT_DIR = Path("sae_checkpoints")
BPRNA_PATH = Path("bprna_activations") / f"bprna_layer_{LAYER}.pt"
OUTPUT_DIR = Path("biology_results")
EVAL_BATCH_SIZE = 4096
MIN_FIRINGS_TO_REPORT = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = {
    "S": "Stem", "H": "Hairpin Loop", "I": "Internal Loop",
    "B": "Bulge", "M": "Multiloop", "E": "External Loop", "X": "Ambiguous",
}


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


print(f"Loading SAE: layer {LAYER}, seed {SEED}")
ckpt_path = CHECKPOINT_DIR / f"sae_layer{LAYER}_seed{SEED}.pt"
checkpoint = torch.load(ckpt_path, weights_only=True)
config = checkpoint["config"]

sae = SparseAutoencoder(config["input_dim"], config["dict_size"]).to(DEVICE)
sae.load_state_dict(checkpoint["state_dict"])
sae.eval()

mean = checkpoint["mean"].to(DEVICE)
std = checkpoint["std"].to(DEVICE)
dict_size = config["dict_size"]

print(f"Loading {BPRNA_PATH}...")
bprna_data = torch.load(BPRNA_PATH, weights_only=False)
vectors = bprna_data["vectors"].float()
structural_labels = bprna_data["structural_labels"]
print(f"Total labeled nucleotide vectors: {vectors.shape[0]}")

normalized_cpu = (vectors - mean.cpu()) / (std.cpu() + 1e-6)

overall_counts = Counter(structural_labels)
total_n = len(structural_labels)
print(f"\nOverall dataset label distribution (baseline):")
for label, count in sorted(overall_counts.items(), key=lambda x: -x[1]):
    print(f"  {label} ({LABEL_NAMES.get(label, '?')}): {100*count/total_n:.1f}%")

feature_label_counts = {i: Counter() for i in range(dict_size)}
feature_total_firings = [0] * dict_size

print(f"\nRunning {vectors.shape[0]} vectors through SAE...")
with torch.no_grad():
    for i in range(0, normalized_cpu.shape[0], EVAL_BATCH_SIZE):
        batch = normalized_cpu[i:i + EVAL_BATCH_SIZE].to(DEVICE)
        batch_labels = structural_labels[i:i + EVAL_BATCH_SIZE]

        _, features = sae(batch)
        fired = (features > 0).cpu()

        for row_idx in range(fired.shape[0]):
            label = batch_labels[row_idx]
            fired_feature_indices = torch.nonzero(fired[row_idx]).squeeze(-1).tolist()
            if isinstance(fired_feature_indices, int):
                fired_feature_indices = [fired_feature_indices]
            for feat_idx in fired_feature_indices:
                feature_label_counts[feat_idx][label] += 1
                feature_total_firings[feat_idx] += 1

    print(f"  processed {min(i + EVAL_BATCH_SIZE, normalized_cpu.shape[0])}/{normalized_cpu.shape[0]}")

OUTPUT_DIR.mkdir(exist_ok=True)
results = []

for feat_idx in range(dict_size):
    total_firings = feature_total_firings[feat_idx]
    if total_firings < MIN_FIRINGS_TO_REPORT:
        continue

    counts = feature_label_counts[feat_idx]
    top_label, top_count = counts.most_common(1)[0]
    purity = 100 * top_count / total_firings

    results.append({
        "feature_idx": feat_idx,
        "total_firings": total_firings,
        "top_label": top_label,
        "top_label_name": LABEL_NAMES.get(top_label, "?"),
        "purity_pct": round(purity, 1),
        "label_breakdown": dict(counts),
    })

results.sort(key=lambda r: -r["purity_pct"])

print(f"\n{'='*70}")
print(f"Found {len(results)} features with >= {MIN_FIRINGS_TO_REPORT} firings")
print(f"{'='*70}")

print(f"\nTop 15 most STRUCTURALLY PURE features:")
for r in results[:15]:
    print(f"  Feature {r['feature_idx']:5d} | fires {r['total_firings']:5d}x | "
          f"top label: {r['top_label']} ({r['top_label_name']}) | purity: {r['purity_pct']}%")

print(f"\nBottom 15 LEAST pure features (most scattered/noisy):")
for r in results[-15:]:
    print(f"  Feature {r['feature_idx']:5d} | fires {r['total_firings']:5d}x | "
          f"top label: {r['top_label']} ({r['top_label_name']}) | purity: {r['purity_pct']}%")

csv_path = OUTPUT_DIR / f"feature_purity_layer{LAYER}_seed{SEED}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["feature_idx", "total_firings", "top_label", "top_label_name", "purity_pct"])
    for r in results:
        writer.writerow([r["feature_idx"], r["total_firings"], r["top_label"],
                          r["top_label_name"], r["purity_pct"]])

print(f"\nSaved full results to {csv_path}")
