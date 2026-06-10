"""Write the *.meta.json that sits next to every script output."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from rna_sae.config import REPO_ROOT, display_path


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def write_meta(output_path, config, packages=(), extra=None):
    """Record what produced output_path: git commit (and whether tracked files had uncommitted
    changes), the full config, the command line, and versions of the given packages."""
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    meta_path = str(output_path).rsplit(".", 1)[0] + ".meta.json"
    # The output and its meta file may be tracked; rewriting them shouldn't count as uncommitted code.
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                  f":(exclude){display_path(output_path)}", f":(exclude){display_path(meta_path)}")
    meta = {
        "output": display_path(output_path),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "command": sys.argv,
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": versions,
        "config": config,
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path
