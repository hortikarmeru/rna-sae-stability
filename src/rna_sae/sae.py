"""Sparse autoencoders (L1-penalized), training, and checkpoint loading."""
import torch
import torch.nn as nn

from rna_sae.config import data_path

ARCHES = ("l1",)


class SparseAutoencoder(nn.Module):
    """ReLU SAE; sparsity comes from an L1 penalty on the feature activations during training."""

    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def forward(self, x):
        features = self.encode(x)
        return self.decoder(features), features


def build_sae(arch, sae_config):
    if arch == "l1":
        return SparseAutoencoder(sae_config["input_dim"], sae_config["dict_size"])
    raise ValueError(f"arch must be one of {ARCHES}")


def train_sae(arch, sae_config, normalized_vectors, seed, device, log=print):
    """Train one SAE. Everything except the seed is fixed, and the seed drives both the weight
    initialization and the batch order, so differences between seeds come only from randomness."""
    torch.manual_seed(seed)
    sae = build_sae(arch, sae_config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae_config["learning_rate"])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized_vectors),
                                         batch_size=sae_config["batch_size"], shuffle=True)
    epochs = sae_config["epochs"]
    history = []
    for epoch in range(epochs):
        total_recon, total_active, n_batches = 0.0, 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, features = sae(batch)
            recon_loss = nn.functional.mse_loss(reconstruction, batch)
            loss = recon_loss
            if arch == "l1":
                loss = loss + sae_config["l1"]["l1_coeff"] * features.abs().sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_active += (features > 0).float().sum(dim=1).mean().item()
            n_batches += 1
        row = {"epoch": epoch + 1, "recon_loss": total_recon / n_batches,
               "avg_active_features": total_active / n_batches}
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            log(f"  seed {seed} | epoch {epoch + 1:2d}/{epochs} | recon_loss {row['recon_loss']:.4f} | "
                f"active {row['avg_active_features']:.1f}")
    return sae, history


def save_sae(path, sae, arch, sae_config, layer, seed, mean, std, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"arch": arch, "input_dim": sae_config["input_dim"], "dict_size": sae_config["dict_size"],
              "layer": layer, "seed": seed}
    config["l1_coeff"] = sae_config["l1"]["l1_coeff"]
    torch.save({"state_dict": sae.state_dict(), "mean": mean, "std": std, "config": config,
                "history": history}, path)


def load_sae(config, arch, layer, seed, device):
    """Returns (sae, mean, std) for one trained checkpoint, on device, in eval mode."""
    checkpoint = torch.load(data_path(config, "checkpoints", arch=arch, layer=layer, seed=seed),
                            weights_only=True)
    sae = build_sae(arch, config["sae"]).to(device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    return sae, checkpoint["mean"], checkpoint["std"]
