#!/usr/bin/env python
"""Run the pre-specified CUDA inference profile for an S-only release."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.financial.evaluation import build_base_args
from experiments.financial.mamba_rerun import runner_args, selection_params
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.registry import public_name, runtime_model_name, source_model_id
from utils.optuna_support import load_dataset_manifest


PROFILE_MODELS = ("RegimeGatedMamba", "UngatedRegimeLinearMamba", "PatchTST", "TimesNet")
DEFAULT_CONFIG = REPO_ROOT / "configs" / "financial_s_only_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "financial_s_only_retrospective_v1"


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except subprocess.SubprocessError:
        return ""


def _git_status() -> dict:
    try:
        lines = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        ).splitlines()
        return {
            "dirty": bool(lines),
            "changed_path_count": len(lines),
        }
    except subprocess.SubprocessError:
        return {"dirty": None, "changed_path_count": None}


def _profile_task(protocol: dict) -> str:
    if "tasks" in protocol:
        return "financial_cross_asset_s"
    return "financial_cross_asset_gld_s"


def _profile_one(
    model_id: str,
    dataset_cfg: dict,
    config: dict,
    cli: Namespace,
    warmup: int,
    repeats: int,
    iterations: int,
) -> dict:
    import torch

    params, selection_path = selection_params(config, dataset_cfg["slug"], model_id)
    args = build_base_args(cli, dataset_cfg, runtime_model_name(model_id), params, 60, "s_only_microbenchmark")
    # Fixed single-target workload; only model architecture/hyperparameters vary.
    args.seq_len = 96
    args.label_len = 48
    args.batch_size = 32
    experiment = Exp_Long_Term_Forecast(args)
    model = experiment.model.eval()
    inputs = torch.randn(args.batch_size, args.seq_len, args.enc_in, device=experiment.device)
    decoder = torch.zeros(args.batch_size, args.label_len + args.pred_len, args.dec_in, device=experiment.device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs, None, decoder, None)
        torch.cuda.synchronize(experiment.device)
        timings = []
        torch.cuda.reset_peak_memory_stats(experiment.device)
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(iterations):
                model(inputs, None, decoder, None)
            torch.cuda.synchronize(experiment.device)
            timings.append(1000.0 * (time.perf_counter() - started) / iterations)
    row = {
        "model_id": model_id,
        "source_model_id": source_model_id(model_id),
        "model": public_name(model_id),
        "task": dataset_cfg["slug"],
        "features": dataset_cfg["features"],
        "horizon": int(args.pred_len),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "warmup_iterations": int(warmup),
        "iterations_per_repeat": int(iterations),
        "repeats": int(repeats),
        "precision": "float32",
        "median_latency_ms": float(np.median(timings)),
        "p10_latency_ms": float(np.quantile(timings, 0.10)),
        "p90_latency_ms": float(np.quantile(timings, 0.90)),
        "throughput_samples_per_second": float(args.batch_size * 1000.0 / np.median(timings)),
        "peak_allocated_memory_mb": float(torch.cuda.max_memory_allocated(experiment.device) / (1024 ** 2)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "selection_source": str(selection_path.relative_to(REPO_ROOT)),
        "gpu": torch.cuda.get_device_name(experiment.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    del experiment, model, inputs, decoder
    torch.cuda.empty_cache()
    return row


def run(args: argparse.Namespace) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched S-only microbenchmark")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads((REPO_ROOT / config["base_protocol"]["path"]).read_text(encoding="utf-8"))
    manifest_path = REPO_ROOT / config["base_protocol"]["training_manifest"]
    task = _profile_task(protocol)
    dataset_cfg = load_dataset_manifest(manifest_path, task, REPO_ROOT)
    cli = runner_args(
        Namespace(
            gpu=args.gpu,
            train_epochs=1,
            patience=1,
            num_workers=0,
            output_root=args.output_root,
            checkpoint_root=args.output_root / "profile_checkpoints",
        ),
        config,
        2023,
    )
    rows = [
        _profile_one(model_id, dataset_cfg, config, cli, args.warmup, args.repeats, args.iterations)
        for model_id in PROFILE_MODELS
    ]
    metrics_root = args.output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    output = metrics_root / "matched_cuda_microbenchmark.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    metadata = {
        "protocol_version": config["protocol_version"],
        "task": task,
        "models": list(PROFILE_MODELS),
        "git_commit": _git_head(),
        "git_status_at_profile": _git_status(),
        "config_sha256": sha256_path(args.config),
        "protocol_sha256": sha256_path(REPO_ROOT / config["base_protocol"]["path"]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "mamba-ssm", "numpy", "pandas")},
        "profile_policy": "fixed S-only single-target CUDA inference workload; no trained checkpoint is loaded",
    }
    atomic_write_json(metrics_root / "matched_cuda_microbenchmark.json", metadata)
    release_root = REPO_ROOT / "reproducibility" / args.release / "metrics"
    release_root.mkdir(parents=True, exist_ok=True)
    (release_root / output.name).write_bytes(output.read_bytes())
    (release_root / "matched_cuda_microbenchmark.json").write_bytes(
        (metrics_root / "matched_cuda_microbenchmark.json").read_bytes()
    )
    return {"status": "complete", "rows": len(rows), "output": str(output.relative_to(REPO_ROOT))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release", choices=("financial_s_only_v1", "financial_s_only_v2"), default="financial_s_only_v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"S-only profile blocked: {exc}") from None
