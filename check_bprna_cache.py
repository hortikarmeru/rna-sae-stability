"""
check_bprna_cache.py

Quick sanity check on the cached bpRNA-90 activations: confirms shapes,
shows the distribution of structural labels, and prints a few example
rows so we can visually confirm everything lines up correctly.

Usage:
    python check_bprna_cache.py
"""

import torch
from pathlib import Path
from collections import Counter

LAYER = 9
path = Path("bprna_activations") / f"bprna_layer_{LAYER}.pt"

print(f"Loading {path}...")
data = torch.load(path, weights_only=False)

vectors = data["vectors"]
sequence_ids = data["sequence_ids"]
positions = data["positions"]
labels = data["structural_labels"]

print(f"\nVectors shape: {vectors.shape}")
print(f"Number of sequence_ids: {len(sequence_ids)}")
print(f"Number of positions: {len(positions)}")
print(f"Number of labels: {len(labels)}")

print(f"\nUnique sequences represented: {len(set(sequence_ids))}")

label_counts = Counter(labels)
print(f"\nStructural label distribution:")
total = len(labels)
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / total
    print(f"  {label}: {count:6d} ({pct:5.1f}%)")

print(f"\nFirst 10 entries (sequence_id, position, label):")
for i in range(10):
    print(f"  {sequence_ids[i]}, pos {positions[i]}, label '{labels[i]}'")
