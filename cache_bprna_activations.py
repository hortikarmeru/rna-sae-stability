"""
cache_bprna_activations.py

Loads RiNALMo, runs sequences from the bpRNA-90 dataset (which includes
per-nucleotide structural annotations: Stem/Hairpin/Internal loop/Bulge/
Multiloop/External loop/Ambiguous/Pseudoknot), and saves token-level
hidden states from layers 9 and 18 -- WITH each vector tagged to its
source sequence ID, position, and structural label.

This is different from cache_activations.py (which threw away all
positional/structural info) -- this version is for biological
interpretation, so traceability per-nucleotide is the whole point.

Usage:
    python cache_bprna_activations.py
"""

import torch
from datasets import load_dataset
from multimolecule import RnaTokenizer, RiNALMoModel
from pathlib import Path

# ---- Config -----------------------------------------------------------
N_SEQUENCES = 500          # bpRNA-90 sample size for interpretation
BATCH_SIZE = 8
LAYERS_TO_SAVE = [9, 18]
MAX_LENGTH = 512
OUTPUT_DIR = Path("bprna_activations")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Setup --------------------------------------------------------------
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rna")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo-giga")
model.to(DEVICE)
model.eval()
model.half()

# ---- Load bpRNA-90 data ---------------------------------------------------
print("Loading bpRNA-90 dataset...")
dataset = load_dataset("multimolecule/bprna-90", split="train", streaming=True)

# Each kept example: (id, sequence, structural_annotation string)
examples = []
for example in dataset:
    seq = example["sequence"]
    if len(seq) <= MAX_LENGTH:
        examples.append({
            "id": example["id"],
            "sequence": seq,
            "structural_annotation": example["structural_annotation"],
        })
    if len(examples) >= N_SEQUENCES:
        break

print(f"Collected {len(examples)} sequences (max length {MAX_LENGTH})")

# Sanity check: structural_annotation should be the same length as sequence
mismatches = sum(
    1 for ex in examples if len(ex["sequence"]) != len(ex["structural_annotation"])
)
print(f"Sequences with annotation length mismatch: {mismatches} / {len(examples)}")

# ---- Run batches, cache activations WITH metadata --------------------------
cache = {
    layer: {
        "vectors": [],
        "sequence_ids": [],
        "positions": [],
        "structural_labels": [],
    }
    for layer in LAYERS_TO_SAVE
}

with torch.no_grad():
    for batch_start in range(0, len(examples), BATCH_SIZE):
        batch = examples[batch_start:batch_start + BATCH_SIZE]
        batch_seqs = [ex["sequence"] for ex in batch]

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs["attention_mask"]

        for layer in LAYERS_TO_SAVE:
            layer_hidden = outputs.hidden_states[layer]

            for i, ex in enumerate(batch):
                real_len = attention_mask[i].sum().item()
                seq_vectors = layer_hidden[i, 1:real_len - 1, :].cpu()

                n_positions = seq_vectors.shape[0]
                labels = ex["structural_annotation"]

                n_use = min(n_positions, len(labels))

                for pos in range(n_use):
                    cache[layer]["vectors"].append(seq_vectors[pos])
                    cache[layer]["sequence_ids"].append(ex["id"])
                    cache[layer]["positions"].append(pos)
                    cache[layer]["structural_labels"].append(labels[pos])

        print(f"  processed {batch_start + len(batch)}/{len(examples)}")

# ---- Save -----------------------------------------------------------------
for layer in LAYERS_TO_SAVE:
    stacked_vectors = torch.stack(cache[layer]["vectors"])

    out_path = OUTPUT_DIR / f"bprna_layer_{layer}.pt"
    torch.save({
        "vectors": stacked_vectors,
        "sequence_ids": cache[layer]["sequence_ids"],
        "positions": cache[layer]["positions"],
        "structural_labels": cache[layer]["structural_labels"],
    }, out_path)

    print(f"Saved {stacked_vectors.shape[0]} labeled nucleotide vectors to {out_path}")

print("Done.")
