"""
train_sae_multiseed.py

Trains multiple SAEs on the same cached activations, differing ONLY by
random seed. This is the core data-generation step for the feature
stability ablation: once these checkpoints exist, a separate script will
compare decoder weight columns across seeds to measure how many features
reproduce reliably.

All hyperparameters below are FIXED across every seed -- only the seed
itself changes. This is intentional and important: any difference in
the resulting SAEs should be attributable to random initialization,
not to a hyperparameter change.

Usage:
    python train_sae_multiseed.py
"""

import torch
import torch.nn as nn
from pathlib import Path

# ---- Config (FIXED across all seeds and layers) -----------------------
LAYERS = [9, 18]         # train on both cached layers
OUTPUT_DIR = Path("sae_checkpoints")

INPUT_DIM = 1280
DICT_SIZE = 8192
L1_COEFF = 1.4e-2        # locked in after tuning -- do not change between seeds
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 40

SEEDS = list(range(10))  # 10 seeds per layer -> 45 pairwise comparisons each

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---- The SAE model (identical to train_sae.py) -----------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


def train_one_seed(layer, seed, normalized_vectors, mean, std):
    """Trains a single SAE with the given layer + seed. Returns nothing;
    saves a checkpoint to disk."""

    torch.manual_seed(seed)

    sae = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

    dataset = torch.utils.data.TensorDataset(normalized_vectors)
    # shuffle uses the global torch RNG, which we just seeded above, so
    # the batch order itself also differs by seed -- this is intentional,
    # it's part of what "differing only by random seed" means in practice.
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(N_EPOCHS):
        total_recon_loss = 0.0
        total_active_features = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            reconstruction, features = sae(batch)

            recon_loss = nn.functional.mse_loss(reconstruction, batch)
            l1_loss = features.abs().sum(dim=1).mean()
            loss = recon_loss + L1_COEFF * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_active_features += (features > 0).float().sum(dim=1).mean().item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            avg_recon = total_recon_loss / n_batches
            avg_active = total_active_features / n_batches
            print(
                f"  layer {layer} seed {seed} | epoch {epoch+1:2d}/{N_EPOCHS} | "
                f"recon_loss: {avg_recon:.4f} | "
                f"avg active features: {avg_active:.1f} / {DICT_SIZE}"
            )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"sae_layer{layer}_seed{seed}.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "mean": mean,
        "std": std,
        "config": {
            "input_dim": INPUT_DIM,
            "dict_size": DICT_SIZE,
            "l1_coeff": L1_COEFF,
            "layer": layer,
            "seed": seed,
        },
    }, out_path)
    print(f"  Saved {out_path}")


# ---- Run all layers x all seeds --------------------------------------------
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

    # free the raw (un-normalized) copy before training -- not strictly
    # necessary but keeps memory use lower across the long run
    del all_vectors, sequences

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Layer {layer} | Seed {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        train_one_seed(layer, seed, normalized_vectors, mean, std)

    del normalized_vectors

print(f"\nAll done: {len(LAYERS)} layers x {len(SEEDS)} seeds = "
      f"{len(LAYERS)*len(SEEDS)} SAEs saved to {OUTPUT_DIR}/")
