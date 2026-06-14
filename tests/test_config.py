from rna_sae.config import data_path, load_config


def test_config_paths_fill():
    config = load_config()
    path = data_path(config, "checkpoints", arch="l1", layer=9, seed=0)
    assert path.name == "sae_l1_layer9_seed0.pt"
    assert config["analysis"]["target_layer"] in config["model"]["layers"]
