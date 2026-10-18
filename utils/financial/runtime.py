#!/usr/bin/env python
"""Capture the executable environment used for a publication run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "runtime_environment.json"


def version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except subprocess.SubprocessError:
        return None


def capture() -> dict:
    import torch

    devices = []
    for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        devices.append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        })
    return {
        "publication_protocol_version": "financial_reference_evaluation_v1",
        "python": sys.version,
        "executable": Path(sys.executable).name,
        "environment": Path(sys.prefix).name,
        "platform": platform.platform(),
        "git_commit": git_head(),
        "packages": {
            distribution: version(distribution)
            for distribution in (
                "torch", "mamba-ssm", "causal-conv1d", "numpy", "pandas",
                "scipy", "scikit-learn", "optuna", "statsmodels", "einops",
                "reformer-pytorch", "yfinance",
            )
        },
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "devices": devices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
