"""Seeding, environment capture, and run manifests."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seed_everything(seed: int) -> np.random.Generator:
    """Seed numpy/torch deterministically; return a fresh Generator."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    import random
    random.seed(seed)
    return np.random.default_rng(seed)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    for pkg in ("numpy", "scipy", "pandas", "cvxpy", "stable_baselines3",
                "gymnasium", "statsmodels"):
        try:
            info[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = None
    return info


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(path: str, config: Dict[str, Any], seeds: Dict[str, Any],
                       extra: Optional[Dict[str, Any]] = None,
                       command: Optional[str] = None) -> Dict[str, Any]:
    manifest = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": command or " ".join(sys.argv),
        "environment": environment_info(),
        "seeds": seeds,
        "config": config,
    }
    if extra:
        manifest.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def finalize_run_manifest(path: str, status: str = "completed") -> None:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["exit_status"] = status
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
