"""
train_sae_topk.py

TopK SAE: instead of an L1 sparsity penalty, directly keeps only the K
largest activations per input and zeros out everything else. This
guarantees sparsity by construction rather than hoping a penalty term
gets there -- the goal is to avoid the collapse we saw with the L1
version, where features either died completely or became near-universal
(firing on 70-85% of all inputs).

Same data, same dictionary size, same layer as the L1 version -- the
ONLY thing changing is the sparsity mechanism, so this is a fair
like-for-like comparison.

Usage:
    python train_sae_topk.py
"""

import torch
import torch.nn as nn
from pathlib import Path

# ---- Config -------------------------------------------------------------
LAYER = 9
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"
OUTPUT_DIR = Path("sae_checkpoints_topk")

INPUT_DIM = 1280
DICT_SIZE = 8192
K = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 40
SEED = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def forward(self, x):
        pre_activations = self.encoder(x)

        topk_values, topk_indices = torch.topk(pre_activations, self.k, dim=1)
        topk_values = torch.relu(topk_values)

        features = torch.zeros_like(pre_activations)
        features.scatter_(1, topk_indices, topk_values)

        reconstruction = self.decoder(features)
        return reconstruction, features


print(f"Loading activations from {ACTIVATIONS_PATH}...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)
all_vectors = torch.cat(sequences, dim=0).float()
print(f"Total token vectors: {all_vectors.shape[0]}")

mean = all_vectors.mean(dim=0, keepdim=True)
std = all_vectors.std(dim=0, keepdim=True)
normalized_vectors = (all_vectors - mean) / (std + 1e-6)

sae = TopKSparseAutoencoder(INPUT_DIM, DICT_SIZE, K).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

dataset = torch.utils.data.TensorDataset(normalized_vectors)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Training TopK SAE (K={K}) on {DEVICE} for {N_EPOCHS} epochs...")

for epoch in range(N_EPOCHS):
    total_recon_loss = 0.0
    total_active_features = 0.0
    n_batches = 0

    for (batch,) in loader:
        batch = batch.to(DEVICE)

        reconstruction, features = sae(batch)
        loss = nn.functional.mse_loss(reconstruction, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon_loss += loss.item()
        total_active_features += (features > 0).float().sum(dim=1).mean().item()
        n_batches += 1

    avg_recon = total_recon_loss / n_batches
    avg_active = total_active_features / n_batches

    print(
        f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
        f"recon_loss: {avg_recon:.4f} | "
        f"avg active features: {avg_active:.1f} (target K={K})"
    )

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / f"sae_topk_layer{LAYER}_seed{SEED}.pt"
torch.save({
    "state_dict": sae.state_dict(),
    "mean": mean,
    "std": std,
    "config": {
        "input_dim": INPUT_DIM,
        "dict_size": DICT_SIZE,
        "k": K,
        "layer": LAYER,
        "seed": SEED,
    },
}, out_path)

print(f"Saved TopK SAE to {out_path}")
