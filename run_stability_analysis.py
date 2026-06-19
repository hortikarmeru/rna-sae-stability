"""
run_stability_analysis.py

Full stability ablation: for each layer, loads all 10 seed checkpoints,
computes each one's alive features, then runs every pairwise comparison
(45 pairs per layer) via Hungarian-matched cosine similarity of decoder
columns. Saves per-pair summary stats and full matched-similarity lists
to CSV for later analysis/plotting.

Usage:
    python run_stability_analysis.py
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from itertools import combinations
import csv

# ---- Config -------------------------------------------------------------
LAYERS = [9, 18]
SEEDS = list(range(10))
CHECKPOINT_DIR = Path("sae_checkpoints")
OUTPUT_DIR = Path("stability_results")
COSINE_THRESHOLD = 0.7
EVAL_BATCH_SIZE = 4096

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


def load_sae(layer, seed):
    path = CHECKPOINT_DIR / f"sae_layer{layer}_seed{seed}.pt"
    checkpoint = torch.load(path, weights_only=True)
    config = checkpoint["config"]
    sae = SparseAutoencoder(config["input_dim"], config["dict_size"]).to(DEVICE)
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


def compare_pair(decoder_a, decoder_b):
    """Returns sorted (descending) matched cosine similarities for one
    pair of (already alive-filtered, normalized) decoder matrices."""
    with torch.no_grad():
        similarity_matrix = decoder_a.T @ decoder_b
        cost_matrix = (-similarity_matrix).cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_similarities = similarity_matrix[row_indices, col_indices].cpu()
    return matched_similarities


# ---- Main ------------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
summary_rows = []

for layer in LAYERS:
    print(f"\n{'#'*60}")
    print(f"# LAYER {layer}")
    print(f"{'#'*60}")

    activations_path = Path("activations") / f"layer_{layer}.pt"
    print(f"Loading activations from {activations_path}...")
    sequences = torch.load(activations_path, weights_only=True)
    all_vectors_cpu = torch.cat(sequences, dim=0).float()
    del sequences
    print(f"Total token vectors: {all_vectors_cpu.shape[0]}")

    # ---- Precompute alive-feature decoder matrices for every seed -----
    decoders = {}    # seed -> normalized decoder matrix (alive features only)
    n_alive = {}      # seed -> count, for reporting

    for seed in SEEDS:
        print(f"  Preparing seed {seed}...")
        sae, mean, std, config = load_sae(layer, seed)
        dict_size = config["dict_size"]

        normalized_cpu = (all_vectors_cpu - mean.cpu()) / (std.cpu() + 1e-6)
        alive_mask = find_alive_features(sae, normalized_cpu, dict_size)

        decoders[seed] = get_normalized_decoder(sae, alive_mask)
        n_alive[seed] = alive_mask.sum().item()

        del sae, normalized_cpu
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    del all_vectors_cpu

    print(f"\n  Alive feature counts: {n_alive}")

    # ---- All pairwise comparisons --------------------------------------
    pairs = list(combinations(SEEDS, 2))
    print(f"  Running {len(pairs)} pairwise comparisons...")

    all_matched_sims_this_layer = []

    for seed_a, seed_b in pairs:
        matched_sims = compare_pair(decoders[seed_a], decoders[seed_b])

        n_stable = (matched_sims >= COSINE_THRESHOLD).sum().item()
        n_matched = len(matched_sims)
        pct_stable = 100 * n_stable / n_matched if n_matched > 0 else 0.0
        mean_sim = matched_sims.mean().item() if n_matched > 0 else 0.0

        summary_rows.append({
            "layer": layer,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "n_alive_a": n_alive[seed_a],
            "n_alive_b": n_alive[seed_b],
            "n_matched": n_matched,
            "n_stable": n_stable,
            "pct_stable": round(pct_stable, 2),
            "mean_matched_similarity": round(mean_sim, 4),
        })

        all_matched_sims_this_layer.extend(matched_sims.tolist())

    avg_pct_stable = sum(r["pct_stable"] for r in summary_rows if r["layer"] == layer) / len(pairs)
    print(f"\n  Layer {layer} summary: average {avg_pct_stable:.1f}% of features stable across {len(pairs)} pairs")

    # Save raw matched similarities for this layer (useful for histograms later)
    raw_path = OUTPUT_DIR / f"matched_similarities_layer{layer}.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["similarity"])
        for sim in all_matched_sims_this_layer:
            writer.writerow([sim])
    print(f"  Saved raw similarities to {raw_path}")

# ---- Save summary table -----------------------------------------------------
summary_path = OUTPUT_DIR / "pairwise_summary.csv"
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\n{'='*60}")
print(f"All done. Summary saved to {summary_path}")
print(f"{'='*60}")

for layer in LAYERS:
    layer_rows = [r for r in summary_rows if r["layer"] == layer]
    avg = sum(r["pct_stable"] for r in layer_rows) / len(layer_rows)
    print(f"Layer {layer}: average {avg:.1f}% stable across {len(layer_rows)} seed pairs")
