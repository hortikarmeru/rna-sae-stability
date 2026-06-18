"""
cache_activations.py

Loads RiNALMo, runs a sample of RNAcentral sequences through it, and
saves token-level hidden states from layers 9 and 18 to disk as .pt files.

Usage:
    python cache_activations.py
"""

import multimolecule  # must import before from_pretrained calls; registers RiNALMo with transformers
import torch
from datasets import load_dataset
from multimolecule import RnaTokenizer, RiNALMoModel
from pathlib import Path

# ---- Config -----------------------------------------------------------
N_SEQUENCES = 300          # how many sequences to pull for this test run
BATCH_SIZE = 8             # sequences per forward pass
LAYERS_TO_SAVE = [9, 18]   # which transformer block outputs to keep
MAX_LENGTH = 512           # truncate/skip sequences longer than this (nucleotides)
OUTPUT_DIR = Path("activations")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Setup --------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo")
model.to(DEVICE)
model.eval()
model.half()  # fp16

# ---- Load data ----------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("multimolecule/rnacentral.1024", split="train", streaming=True)

sequences = []
for example in dataset:
    seq = example["sequence"]
    if len(seq) <= MAX_LENGTH:
        sequences.append(seq)
    if len(sequences) >= N_SEQUENCES:
        break

print(f"Collected {len(sequences)} sequences (max length {MAX_LENGTH})")

# ---- Run batches and cache activations -----------------------------------
cache = {layer: [] for layer in LAYERS_TO_SAVE}

with torch.no_grad():
    for batch_start in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[batch_start:batch_start + BATCH_SIZE]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs["attention_mask"]

        for layer in LAYERS_TO_SAVE:
            layer_hidden = outputs.hidden_states[layer]

            for i in range(len(batch_seqs)):
                real_len = attention_mask[i].sum().item()
                seq_activations = layer_hidden[i, 1:real_len - 1, :].cpu()
                cache[layer].append(seq_activations)

        print(f"  processed {batch_start + len(batch_seqs)}/{len(sequences)}")

# ---- Save -----------------------------------------------------------------
for layer in LAYERS_TO_SAVE:
    out_path = OUTPUT_DIR / f"layer_{layer}.pt"
    torch.save(cache[layer], out_path)
    print(f"Saved {len(cache[layer])} sequences to {out_path}")

print("Done.")
