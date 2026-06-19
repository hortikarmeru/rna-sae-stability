"""
check_topk_features.py

Same diagnostic as check_dead_features.py, but for the TopK SAE, plus
the firing-rate check that revealed the "universal firing" collapse in
the L1 version. Checks:
  1. How many of the 8192 features ever fire at all (alive vs dead)
  2. For features that DO fire, what fraction of all inputs do they fire
     on -- this is the check that matters most, since the L1 version's
     alive features were firing on 70-85% of everything (not selective).

Usage:
    python check_topk_features.py
"""

import torch
import torch.nn as nn
from pathlib import Path

LAYER = 9
SEED = 0
CHECKPOINT_DIR = Path("sae_checkpoints_topk")
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


path = CHECKPOINT_DIR / f"sae_topk_layer{LAYER}_seed{SEED}.pt"
checkpoint = torch.load(path, weights_only=True)
config = checkpoint["config"]

sae = TopKSparseAutoencoder(config["input_dim"], config["dict_size"], config["k"]).to(DEVICE)
sae.load_state_dict(checkpoint["state_dict"])
sae.eval()

mean = checkpoint["mean"].to(DEVICE)
std = checkpoint["std"].to(DEVICE)
dict_size = config["dict_size"]
k = config["k"]

print(f"Loading activations from {ACTIVATIONS_PATH}...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)
all_vectors = torch.cat(sequences, dim=0).float()
normalized_vectors_cpu = (all_vectors - mean.cpu()) / (std.cpu() + 1e-6)
n_total = normalized_vectors_cpu.shape[0]
print(f"Total token vectors: {n_total}")

BATCH_SIZE = 4096
firing_counts = torch.zeros(dict_size, device=DEVICE)

with torch.no_grad():
    for i in range(0, n_total, BATCH_SIZE):
        batch = normalized_vectors_cpu[i:i + BATCH_SIZE].to(DEVICE)
        _, features = sae(batch)
        fired = features > 0
        firing_counts += fired.float().sum(dim=0)

firing_counts_cpu = firing_counts.cpu()
firing_rates = firing_counts_cpu / n_total

n_alive = (firing_counts_cpu > 0).sum().item()
n_dead = dict_size - n_alive

print(f"\n--- TopK feature check: layer {LAYER}, seed {SEED}, K={k} ---")
print(f"Total features: {dict_size}")
print(f"Alive (fired at least once): {n_alive} ({100*n_alive/dict_size:.1f}%)")
print(f"Dead (never fired): {n_dead} ({100*n_dead/dict_size:.1f}%)")

alive_rates = firing_rates[firing_counts_cpu > 0]
if len(alive_rates) > 0:
    print(f"\nFiring rate stats among ALIVE features (fraction of {n_total} inputs each fires on):")
    print(f"  Min:    {alive_rates.min().item()*100:.2f}%")
    print(f"  Median: {alive_rates.median().item()*100:.2f}%")
    print(f"  Mean:   {alive_rates.mean().item()*100:.2f}%")
    print(f"  Max:    {alive_rates.max().item()*100:.2f}%")

    n_selective = (alive_rates < 0.10).sum().item()
    n_moderate = ((alive_rates >= 0.10) & (alive_rates < 0.50)).sum().item()
    n_universal = (alive_rates >= 0.50).sum().item()

    print(f"\nSelectivity breakdown (of {n_alive} alive features):")
    print(f"  Selective (fire on <10% of inputs):  {n_selective} ({100*n_selective/n_alive:.1f}%)")
    print(f"  Moderate (fire on 10-50% of inputs):  {n_moderate} ({100*n_moderate/n_alive:.1f}%)")
    print(f"  Universal (fire on 50%+ of inputs):   {n_universal} ({100*n_universal/n_alive:.1f}%)")
