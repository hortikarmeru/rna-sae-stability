"""
train_sae_topk_multiseed.py

Trains TopK SAEs across both layers x 10 seeds, mirroring the structure
of train_sae_multiseed.py (the L1 version) for a direct, fair comparison.
Same K=32 for both layers deliberately, to avoid the per-layer-tuning
confound that affected the L1 layer comparison.

Usage:
    python train_sae_topk_multiseed.py
"""

import torch
import torch.nn as nn
from pathlib import Path

# ---- Config (FIXED across all seeds and layers) -----------------------
LAYERS = [9, 18]
OUTPUT_DIR = Path("sae_checkpoints_topk")

INPUT_DIM = 1280
DICT_SIZE = 8192
K = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 40

SEEDS = list(range(10))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def train_one_seed(layer, seed, normalized_vectors, mean, std):
    torch.manual_seed(seed)

    sae = TopKSparseAutoencoder(INPUT_DIM, DICT_SIZE, K).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

    dataset = torch.utils.data.TensorDataset(normalized_vectors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(N_EPOCHS):
        total_recon_loss = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            reconstruction, features = sae(batch)
            loss = nn.functional.mse_loss(reconstruction, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            avg_recon = total_recon_loss / n_batches
            print(f"  layer {layer} seed {seed} | epoch {epoch+1:2d}/{N_EPOCHS} | "
                  f"recon_loss: {avg_recon:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"sae_topk_layer{layer}_seed{seed}.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "mean": mean,
        "std": std,
        "config": {
            "input_dim": INPUT_DIM,
            "dict_size": DICT_SIZE,
            "k": K,
            "layer": layer,
            "seed": seed,
        },
    }, out_path)
    print(f"  Saved {out_path}")


for layer in LAYERS:
    print(f"\n{'#'*60}")
    print(f"# LAYER {layer}")
    print(f"{'#'*60}")

    activations_path = Path("activations") / f"layer_{layer}.pt"
    print(f"Loading activations from {activations_path}...")
    sequences = torch.load(activations_path, weights_only=True)
    all_vectors = torch.cat(sequences, dim=0).float()
    print(f"Total token vectors: {all_vectors.shape[0]}")

    mean = all_vectors.mean(dim=0, keepdim=True)
    std = all_vectors.std(dim=0, keepdim=True)
    normalized_vectors = (all_vectors - mean) / (std + 1e-6)

    del all_vectors, sequences

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Layer {layer} | Seed {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        train_one_seed(layer, seed, normalized_vectors, mean, std)

    del normalized_vectors

print(f"\nAll done: {len(LAYERS)} layers x {len(SEEDS)} seeds = "
      f"{len(LAYERS)*len(SEEDS)} TopK SAEs saved to {OUTPUT_DIR}/")
