"""Feature firing statistics and cross-seed matching of SAE features."""
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def firing_counts(sae, normalized_vectors, batch_size, device):
    """How many of the input vectors each dictionary feature fires on (activation > 0)."""
    counts = torch.zeros(sae.decoder.weight.shape[1], device=device)
    for i in range(0, normalized_vectors.shape[0], batch_size):
        _, features = sae(normalized_vectors[i:i + batch_size].to(device))
        counts += (features > 0).float().sum(dim=0)
    return counts.cpu()


def unit_decoder(sae):
    """Decoder columns scaled to unit length (one column per feature)."""
    weight = sae.decoder.weight
    return weight / (weight.norm(dim=0, keepdim=True) + 1e-8)


def match_features(decoder_a, decoder_b):
    """One-to-one matching of features between two SAEs that maximizes total cosine similarity
    (Hungarian algorithm). Returns (rows, cols, similarities) with rows/cols indexing the
    features of a and b."""
    similarity = decoder_a.T @ decoder_b
    rows, cols = linear_sum_assignment((-similarity).cpu().numpy())
    return rows, cols, similarity[rows, cols].cpu()
