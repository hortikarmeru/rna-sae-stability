"""
train_sae.py

Trains a single sparse autoencoder (SAE) on cached RiNALMo activations
from one layer. This is a TEST version meant to run on the small
(300-sequence) activation cache, just to verify the training loop works
correctly before scaling up to a real dataset.

Usage:
    python train_sae.py
"""

import torch
import torch.nn as nn
from pathlib import Path

# ---- Config -----------------------------------------------------------
LAYER = 9
ACTIVATIONS_PATH = Path("activations") / f"layer_{LAYER}.pt"
OUTPUT_DIR = Path("sae_checkpoints")

INPUT_DIM = 1280       # RiNALMo's hidden size, fixed by the model
DICT_SIZE = 8192        # number of SAE features (overcomplete: 8192 > 1280)
L1_COEFF = 1e-3          # sparsity penalty strength (lambda)
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
N_EPOCHS = 20
SEED = 0                # random seed -- this is what we'll vary across SAEs later

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Reproducibility ----------------------------------------------------
torch.manual_seed(SEED)

# ---- The SAE model --------------------------------------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        # Untied weights: encoder and decoder are separate matrices.
        # This matches the SAE-RNA paper's design (not a tied-weight autoencoder).
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

        # Decoder columns are conventionally initialized as unit-norm vectors;
        # this keeps early training stable since each feature "direction"
        # starts the same length.
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def forward(self, x):
        # ReLU enforces non-negativity, which combined with L1 loss
        # produces sparsity: most hidden units end up exactly at 0.
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features


# ---- Load data ------------------------------------------------------------
print(f"Loading activations from {ACTIVATIONS_PATH}...")
sequences = torch.load(ACTIVATIONS_PATH, weights_only=True)

# Each entry is (L, 1280) for one sequence, L varies. The SAE doesn't care
# about sequence structure or position -- it just wants individual
# 1280-dim vectors. So we concatenate every token from every sequence
# into one big (N, 1280) tensor.
all_vectors = torch.cat(sequences, dim=0)
print(f"Total token vectors: {all_vectors.shape[0]}")

# Cast to float32 for training -- fp16 storage was for saving disk space,
# but training is more numerically stable in float32.
all_vectors = all_vectors.float()

# ---- Standardize ------------------------------------------------------
# Subtract mean, divide by std, per-dimension. SAE interpretability work
# generally trains on standardized activations so no single dimension
# dominates the reconstruction loss just because it has larger raw scale.
mean = all_vectors.mean(dim=0, keepdim=True)
std = all_vectors.std(dim=0, keepdim=True)
normalized_vectors = (all_vectors - mean) / (std + 1e-6)

# ---- Setup ----------------------------------------------------------------
sae = SparseAutoencoder(INPUT_DIM, DICT_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

dataset = torch.utils.data.TensorDataset(normalized_vectors)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Training loop ----------------------------------------------------------
print(f"Training on {DEVICE} for {N_EPOCHS} epochs...")

for epoch in range(N_EPOCHS):
    total_recon_loss = 0.0
    total_l1_loss = 0.0
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
        total_l1_loss += l1_loss.item()
        # average number of nonzero features per example in this batch
        total_active_features += (features > 0).float().sum(dim=1).mean().item()
        n_batches += 1

    avg_recon = total_recon_loss / n_batches
    avg_l1 = total_l1_loss / n_batches
    avg_active = total_active_features / n_batches

    print(
        f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
        f"recon_loss: {avg_recon:.4f} | "
        f"l1_loss: {avg_l1:.2f} | "
        f"avg active features: {avg_active:.1f} / {DICT_SIZE}"
    )

# ---- Save -----------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / f"sae_layer{LAYER}_seed{SEED}.pt"
torch.save({
    "state_dict": sae.state_dict(),
    "mean": mean,
    "std": std,
    "config": {
        "input_dim": INPUT_DIM,
        "dict_size": DICT_SIZE,
        "l1_coeff": L1_COEFF,
        "layer": LAYER,
        "seed": SEED,
    },
}, out_path)

print(f"Saved SAE to {out_path}")
