"""
compare_stability_test.py

TEST VERSION: compares exactly one pair of SAEs (same layer, two seeds)
to verify the stability-measurement mechanics work correctly before
running the full 45-pair comparison.

Usage:
    python compare_stability_test.py
"""

import torch
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# ---- Config -------------------------------------------------------------
LAYER = 9
SEED_A = 0
SEED_B = 1
CHECKPOINT_DIR = Path("sae_checkpoints")
COSINE_THRESHOLD = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_decoder_weights(layer, seed):
    """Loads one SAE checkpoint and returns its decoder weight matrix,
    with each column normalized to unit length (so cosine similarity
    is just a dot product)."""
    path = CHECKPOINT_DIR / f"sae_layer{layer}_seed{seed}.pt"
    checkpoint = torch.load(path, weights_only=True)

    # The decoder is an nn.Linear(dict_size, input_dim, bias=False).
    # PyTorch stores Linear weight as (out_features, in_features), i.e.
    # (input_dim, dict_size) here -- so each COLUMN is one feature's
    # direction in the original 1280-dim activation space.
    decoder_weight = checkpoint["state_dict"]["decoder.weight"]  # (1280, 8192)

    # Normalize each column to unit length.
    norms = decoder_weight.norm(dim=0, keepdim=True)  # (1, 8192)
    normalized = decoder_weight / (norms + 1e-8)

    return normalized.to(DEVICE)


print(f"Loading decoder weights: layer {LAYER}, seed {SEED_A} vs seed {SEED_B}")
decoder_a = load_decoder_weights(LAYER, SEED_A)  # (1280, 8192)
decoder_b = load_decoder_weights(LAYER, SEED_B)  # (1280, 8192)

print(f"decoder_a shape: {decoder_a.shape}")
print(f"decoder_b shape: {decoder_b.shape}")

# ---- Cosine similarity matrix -----------------------------------------
# Since both decoders' columns are already unit-normalized, cosine
# similarity between feature i (in A) and feature j (in B) is simply
# their dot product. Matrix multiplying (8192, 1280) x (1280, 8192)
# gives us all pairwise similarities at once: an (8192, 8192) matrix
# where entry [i, j] = cosine_similarity(feature_i_A, feature_j_B).
print("Computing cosine similarity matrix...")
similarity_matrix = decoder_a.T @ decoder_b  # (8192, 8192)
print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Similarity range: [{similarity_matrix.min().item():.3f}, {similarity_matrix.max().item():.3f}]")

# ---- Hungarian matching -------------------------------------------------
# linear_sum_assignment finds the matching that MINIMIZES total cost, so
# we feed it negative similarity (maximizing similarity = minimizing
# negative similarity). It requires a numpy array on CPU.
print("Running Hungarian algorithm (this may take a moment)...")
cost_matrix = (-similarity_matrix).cpu().numpy()
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# row_indices[i] is matched to col_indices[i]; since the matrix is square
# (8192x8192) and we're not subsetting, row_indices will just be
# [0, 1, 2, ..., 8191] in order -- but col_indices tells us, for each
# feature in A, which feature in B it was matched to.
matched_similarities = similarity_matrix[row_indices, col_indices].cpu()

# ---- Results --------------------------------------------------------------
n_stable = (matched_similarities >= COSINE_THRESHOLD).sum().item()
n_total = len(matched_similarities)
pct_stable = 100 * n_stable / n_total

print(f"\n--- Results: layer {LAYER}, seed {SEED_A} vs seed {SEED_B} ---")
print(f"Total features: {n_total}")
print(f"Features with matched cosine similarity >= {COSINE_THRESHOLD}: {n_stable}")
print(f"Percent stable: {pct_stable:.1f}%")
print(f"Mean matched similarity: {matched_similarities.mean().item():.3f}")
print(f"Median matched similarity: {matched_similarities.median().item():.3f}")

# Sanity check: print a few example matched similarities
print(f"\nFirst 10 matched similarities (feature index in A -> best match in B):")
for i in range(10):
    print(f"  A[{i}] <-> B[{col_indices[i]}]: similarity = {matched_similarities[i].item():.3f}")
