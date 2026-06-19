"""
compare_stability_test.py (v2)

TEST VERSION: compares exactly one pair of SAEs (same layer, two seeds).

This version only compares ALIVE features (ones that actually fired at
least once on the training data) -- dead, never-trained features are
excluded entirely, since comparing untrained noise vectors to each
other corrupts the stability measurement.

Usage:
    python compare_stability_test.py
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# ---- Config -------------------------------------------------------------
LAYER = 9
SEED_A = 0
SEED_B = 1
CHECKPOINT_DIR = Path("sae_checkpoints")
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"
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
    decoder_weight = sae.decoder.weight
    alive_weight = decoder_weight[:, alive_mask]
    norms = alive_weight.norm(dim=0, keepdim=True)
    normalized = alive_weight / (norms + 1e-8)
    return normalized


print(f"Loading SAEs: layer {LAYER}, seed {SEED_A} vs seed {SEED_B}")
sae_a, mean_a, std_a, config_a = load_sae(LAYER, SEED_A)
sae_b, mean_b, std_b, config_b = load_sae(LAYER, SEED_B)
dict_size = config_a["dict_size"]

print(f"Loading activations from {ACTIVATIONS_PATH}...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)
all_vectors_cpu = torch.cat(sequences, dim=0).float()
print(f"Total token vectors: {all_vectors_cpu.shape[0]}")

normalized_a_cpu = (all_vectors_cpu - mean_a.cpu()) / (std_a.cpu() + 1e-6)
normalized_b_cpu = (all_vectors_cpu - mean_b.cpu()) / (std_b.cpu() + 1e-6)

print("Finding alive features for seed A...")
alive_a = find_alive_features(sae_a, normalized_a_cpu, dict_size)
print("Finding alive features for seed B...")
alive_b = find_alive_features(sae_b, normalized_b_cpu, dict_size)

n_alive_a = alive_a.sum().item()
n_alive_b = alive_b.sum().item()
print(f"Seed A alive features: {n_alive_a}")
print(f"Seed B alive features: {n_alive_b}")

decoder_a = get_normalized_decoder(sae_a, alive_a)
decoder_b = get_normalized_decoder(sae_b, alive_b)

print(f"Comparing {n_alive_a} alive features (A) against {n_alive_b} alive features (B)...")
similarity_matrix = decoder_a.T @ decoder_b

cost_matrix = (-similarity_matrix).cpu().numpy()
row_indices, col_indices = linear_sum_assignment(cost_matrix)
matched_similarities = similarity_matrix[row_indices, col_indices].cpu()

n_stable = (matched_similarities >= COSINE_THRESHOLD).sum().item()
n_matched = len(matched_similarities)
pct_stable = 100 * n_stable / n_matched if n_matched > 0 else 0.0

print(f"\n--- Results: layer {LAYER}, seed {SEED_A} vs seed {SEED_B} (alive features only) ---")
print(f"Alive features compared: {n_matched} (min of {n_alive_a}, {n_alive_b})")
print(f"Features with matched cosine similarity >= {COSINE_THRESHOLD}: {n_stable}")
print(f"Percent stable: {pct_stable:.1f}%")
print(f"Mean matched similarity: {matched_similarities.mean().item():.3f}")
print(f"Median matched similarity: {matched_similarities.median().item():.3f}")

print(f"\nAll matched similarities, sorted descending:")
sorted_sims, _ = torch.sort(matched_similarities, descending=True)
print(sorted_sims.tolist())
