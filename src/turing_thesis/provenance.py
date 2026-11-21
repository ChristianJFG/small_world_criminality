"""Capture the actual code, numerical environment, and output content."""
import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path
import subprocess
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions() if d.metadata["Name"])),
        "numpy_build": np.show_config(mode="dicts"),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def source_identity():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = list(package.glob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")
              if (root / name).exists()]
    result = {"sha256": {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        result["tracked_changes"] = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result
