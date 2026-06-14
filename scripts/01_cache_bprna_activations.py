"""Cache RiNALMo hidden states for bpRNA-90 sequences, with each nucleotide's structure label.

Usage: python scripts/01_cache_bprna_activations.py

Unlike 00, every vector keeps its sequence ID, position and bpRNA structural annotation
(Stem, Hairpin, Internal loop, Bulge, Multiloop, External loop, Ambiguous), which the
enrichment analysis needs. Saves one dict per layer to paths.bprna_activations.
"""
import torch
from datasets import load_dataset

from rna_sae.activations import hidden_states, load_model
from rna_sae.config import data_path, load_config
from rna_sae.meta import write_meta


def main():
    config = load_config()
    model_cfg, data_cfg = config["model"], config["data"]["bprna"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_cfg, device)

    examples = []
    for ex in load_dataset(data_cfg["dataset"], split="train", streaming=True):
        if len(ex["sequence"]) <= data_cfg["max_length"]:
            examples.append({"id": ex["id"], "sequence": ex["sequence"],
                             "labels": ex["structural_annotation"]})
        if len(examples) >= data_cfg["n_sequences"]:
            break
    print(f"Collected {len(examples)} sequences")

    cache = {layer: {"vectors": [], "sequence_ids": [], "positions": [], "structural_labels": []}
             for layer in model_cfg["layers"]}
    batch_size = model_cfg["batch_size"]
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        states = hidden_states(tokenizer, model, [e["sequence"] for e in batch], model_cfg["layers"], device)
        for layer, per_seq in states.items():
            for ex, vectors in zip(batch, per_seq):
                for pos in range(min(vectors.shape[0], len(ex["labels"]))):
                    cache[layer]["vectors"].append(vectors[pos])
                    cache[layer]["sequence_ids"].append(ex["id"])
                    cache[layer]["positions"].append(pos)
                    cache[layer]["structural_labels"].append(ex["labels"][pos])
        print(f"  processed {min(start + batch_size, len(examples))}/{len(examples)}", flush=True)

    for layer, c in cache.items():
        out = data_path(config, "bprna_activations", layer=layer)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**c, "vectors": torch.stack(c["vectors"])}, out)
        write_meta(out, config, packages=["torch", "multimolecule", "datasets"],
                   extra={"n_vectors": len(c["vectors"])})
        print(f"Saved {len(c['vectors'])} labeled nucleotide vectors to {out}")


if __name__ == "__main__":
    main()
