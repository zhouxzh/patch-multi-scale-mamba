#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import (
    DEFAULT_PROTOCOL,
    atomic_write_json,
    load_protocol,
    proxy_is_listening,
    sha256_path,
)


def command_result(command: list[str]) -> dict:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def import_result(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {"ok": True, "version": getattr(module, "__version__", None)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check prerequisites for the financial benchmark.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--allow-no-cuda", action="store_true", help="Return success for data-only work.")
    parser.add_argument("--output", type=Path, help="Optional path for an atomic JSON environment report.")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)

    try:
        import torch

        cuda = {
            "torch_version": torch.__version__,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if cuda["available"]:
            cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(cuda["device_count"])]
    except Exception as exc:
        cuda = {"available": False, "device_count": 0, "error": str(exc)}

    proxy_url = protocol["proxy"]["fallback_url"]
    report = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_path(args.protocol),
        "python": sys.version,
        "host": socket.gethostname(),
        "cuda": cuda,
        "nvidia_smi": command_result(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"]),
        "clash_proxy": {"url": proxy_url, "listening": proxy_is_listening(proxy_url)},
        "disk": shutil.disk_usage(Path.cwd())._asdict(),
        "packages": {},
        "runtime_imports": {
            module_name: import_result(module_name)
            for module_name in (
                "mamba_ssm",
                "model.InvertedMamba",
                "model.DecompositionMamba",
                "model.MultiScaleDecompositionMamba",
                "model.RegimeGatedMamba",
            )
        },
        "git": {
            "head": command_result(["git", "rev-parse", "HEAD"]),
            "status": command_result(["git", "status", "--short"]),
        },
    }
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "torch",
        "optuna",
        "mamba-ssm",
        "causal-conv1d",
    ):
        try:
            report["packages"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][distribution] = None
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        atomic_write_json(args.output, report)
        print(f"wrote {args.output}")
    if not cuda["available"] and not args.allow_no_cuda:
        print("CUDA is unavailable; formal neural training is blocked.", file=sys.stderr)
        return 2
    failed_imports = [name for name, result in report["runtime_imports"].items() if not result["ok"]]
    if cuda["available"] and failed_imports:
        print(f"Mamba runtime imports failed: {failed_imports}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
