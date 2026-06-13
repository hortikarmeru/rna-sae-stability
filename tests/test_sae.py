import torch

from rna_sae.features import firing_counts, match_features, unit_decoder
from rna_sae.sae import SparseAutoencoder, train_sae

SAE_CONFIG = {"input_dim": 8, "dict_size": 32, "learning_rate": 1e-3, "batch_size": 16, "epochs": 2,
              "l1": {"l1_coeff": 1e-2}}


def test_decoder_columns_start_at_unit_norm():
    sae = SparseAutoencoder(8, 32)
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(32), atol=1e-5)


def test_matching_recovers_a_permutation():
    sae = SparseAutoencoder(8, 32)
    decoder = unit_decoder(sae, torch.ones(32, dtype=torch.bool))
    perm = torch.randperm(32)
    rows, cols, sims = match_features(decoder, decoder[:, perm])
    assert torch.allclose(sims, torch.ones(32), atol=1e-5)
    assert (perm[cols] == torch.as_tensor(rows)).all()


def test_same_seed_gives_identical_sae():
    data = torch.randn(64, 8)
    a, _ = train_sae("l1", SAE_CONFIG, data, seed=3, device="cpu", log=lambda *_: None)
    b, _ = train_sae("l1", SAE_CONFIG, data, seed=3, device="cpu", log=lambda *_: None)
    assert torch.equal(a.decoder.weight, b.decoder.weight)


def test_firing_counts_shape():
    sae = SparseAutoencoder(8, 32)
    counts = firing_counts(sae, torch.randn(50, 8), batch_size=16, device="cpu")
    assert counts.shape == (32,) and counts.sum() <= 50 * 32
