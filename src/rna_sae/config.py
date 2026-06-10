"""Load configs/config.yaml and build the paths in it."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(p):
    """Resolve a path relative to the repo root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def data_path(config, key, **fields):
    """Fill one of the path templates in config['paths'], e.g. data_path(cfg, 'activations', layer=9)."""
    paths = config["paths"]
    return repo_path(paths[key].format(data_dir=paths["data_dir"], **fields))


def display_path(p):
    """Path relative to the repo root when inside it, so outputs don't record machine-specific paths."""
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
