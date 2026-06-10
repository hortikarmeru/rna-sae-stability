"""Cached RiNALMo activations: extraction, loading and normalization."""
import torch


def load_model(model_config, device):
    from multimolecule import RiNALMoModel, RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(model_config["tokenizer"])
    model = RiNALMoModel.from_pretrained(model_config["name"]).to(device).eval()
    if model_config["dtype"] == "float16":
        model.half()
    return tokenizer, model


@torch.no_grad()
def hidden_states(tokenizer, model, sequences, layers, device):
    """Per-sequence hidden states for each layer, without the start/end tokens.
    Returns {layer: [tensor(len_i, hidden_dim), ...]} on the CPU."""
    inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    lengths = inputs["attention_mask"].sum(dim=1).tolist()
    return {layer: [outputs.hidden_states[layer][i, 1:n - 1, :].cpu() for i, n in enumerate(lengths)]
            for layer in layers}

