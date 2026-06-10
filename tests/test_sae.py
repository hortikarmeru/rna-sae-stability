import torch

from rna_sae.sae import SparseAutoencoder, train_sae

SAE_CONFIG = {"input_dim": 8, "dict_size": 32, "learning_rate": 1e-3, "batch_size": 16, "epochs": 2,
              "l1": {"l1_coeff": 1e-2}}


def test_decoder_columns_start_at_unit_norm():
    sae = SparseAutoencoder(8, 32)
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(32), atol=1e-5)


def test_same_seed_gives_identical_sae():
    data = torch.randn(64, 8)
    a, _ = train_sae("l1", SAE_CONFIG, data, seed=3, device="cpu", log=lambda *_: None)
    b, _ = train_sae("l1", SAE_CONFIG, data, seed=3, device="cpu", log=lambda *_: None)
    assert torch.equal(a.decoder.weight, b.decoder.weight)
