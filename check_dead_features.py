"""
check_dead_features.py

Diagnostic: checks how many of the 8192 SAE features ever activate
(fire nonzero) across the training data, vs how many are "dead"
(never fire, essentially random noise). Dead features would corrupt
the stability comparison, since the Hungarian algorithm is forced to
match every column even if it never learned anything real.

Usage:
    python check_dead_features.py
"""

import torch
import torch.nn as nn
from pathlib import Path

LAYER = 9
SEED = 0
CHECKPOINT_DIR = Path("sae_checkpoints")
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"

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


# ---- Load checkpoint ----------------------------------------------------
path = CHECKPOINT_DIR / f"sae_layer{LAYER}_seed{SEED}.pt"
checkpoint = torch.load(path, weights_only=True)
config = checkpoint["config"]

sae = SparseAutoencoder(config["input_dim"], config["dict_size"]).to(DEVICE)
sae.load_state_dict(checkpoint["state_dict"])
sae.eval()

mean = checkpoint["mean"].to(DEVICE)
std = checkpoint["std"].to(DEVICE)

# ---- Load and normalize data the same way as training -----------------
print(f"Loading activations from {ACTIVATIONS_PATH}...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)
all_vectors = torch.cat(sequences, dim=0).float().to(DEVICE)
normalized_vectors = (all_vectors - mean) / (std + 1e-6)
print(f"Total token vectors: {normalized_vectors.shape[0]}")

# ---- Run through SAE in batches, track which features ever fire -------
BATCH_SIZE = 4096
dict_size = config["dict_size"]
ever_active = torch.zeros(dict_size, dtype=torch.bool, device=DEVICE)
total_activation_count = torch.zeros(dict_size, device=DEVICE)

with torch.no_grad():
    for i in range(0, normalized_vectors.shape[0], BATCH_SIZE):
        batch = normalized_vectors[i:i + BATCH_SIZE]
        _, features = sae(batch)
        fired = features > 0
        ever_active |= fired.any(dim=0)
        total_activation_count += fired.float().sum(dim=0)

n_alive = ever_active.sum().item()
n_dead = dict_size - n_alive

print(f"\n--- Dead feature check: layer {LAYER}, seed {SEED} ---")
print(f"Total features: {dict_size}")
print(f"Alive (fired at least once): {n_alive} ({100*n_alive/dict_size:.1f}%)")
print(f"Dead (never fired): {n_dead} ({100*n_dead/dict_size:.1f}%)")

# How many fired more than just a handful of times (i.e. "real" features
# vs ones that fired by fluke once or twice)?
fired_at_least_10 = (total_activation_count >= 10).sum().item()
fired_at_least_100 = (total_activation_count >= 100).sum().item()
print(f"Fired >= 10 times: {fired_at_least_10} ({100*fired_at_least_10/dict_size:.1f}%)")
print(f"Fired >= 100 times: {fired_at_least_100} ({100*fired_at_least_100/dict_size:.1f}%)")
