"""Cache RiNALMo hidden states for a sample of RNAcentral sequences (SAE training data).

Usage: python scripts/00_cache_activations.py

Streams data.rnacentral.dataset, keeps the first n_sequences within max_length, and saves one
list of per-sequence (length x 1280) tensors per layer in model.layers to paths.activations.
"""
import torch
from datasets import load_dataset

from rna_sae.activations import hidden_states, load_model
from rna_sae.config import data_path, load_config
from rna_sae.meta import write_meta


def main():
    config = load_config()
    model_cfg, data_cfg = config["model"], config["data"]["rnacentral"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(model_cfg, device)

    sequences = []
    for example in load_dataset(data_cfg["dataset"], split="train", streaming=True):
        if len(example["sequence"]) <= data_cfg["max_length"]:
            sequences.append(example["sequence"])
        if len(sequences) >= data_cfg["n_sequences"]:
            break
    print(f"Collected {len(sequences)} sequences (max length {data_cfg['max_length']})")

    cache = {layer: [] for layer in model_cfg["layers"]}
    batch_size = model_cfg["batch_size"]
    for start in range(0, len(sequences), batch_size):
        for layer, states in hidden_states(tokenizer, model, sequences[start:start + batch_size],
                                           model_cfg["layers"], device).items():
            cache[layer].extend(states)
        print(f"  processed {min(start + batch_size, len(sequences))}/{len(sequences)}", flush=True)

    for layer, states in cache.items():
        out = data_path(config, "activations", layer=layer)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states, out)
        write_meta(out, config, packages=["torch", "multimolecule", "datasets"],
                   extra={"n_sequences": len(states), "n_tokens": sum(s.shape[0] for s in states)})
        print(f"Saved {len(states)} sequences to {out}")


if __name__ == "__main__":
    main()
