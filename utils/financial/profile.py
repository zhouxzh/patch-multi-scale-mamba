#!/usr/bin/env python
"""Profile Regime-Gated Mamba, PatchTST, and TimesNet under one CUDA benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.financial.evaluation as runner
from utils.financial.registry import public_name, runtime_model_name, source_model_id
from experiments.financial.reference import (
    load_config,
    load_v3_params,
    runner_args,
)

TASK = "financial_cross_asset_ms"
HORIZON = 60
SEQUENCE_LENGTH = 96
BATCH_SIZE = 32
WARMUP = 20
REPEATS = 5
ITERATIONS = 100
OUTPUT = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "metrics" / "matched_cuda_microbenchmark.csv"


def selected_baseline_params(model: str) -> dict:
    path = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "selection" / TASK / "baseline_best_params.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["models"][model]["params"])


def profile(model_label: str, display_name: str, params: dict, dataset_cfg: dict) -> dict:
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

    cli = runner_args(type("Args", (), {"gpu": 0, "train_epochs": 1, "patience": 1, "num_workers": 0})(), REPO_ROOT / "results" / "financial_reference_evaluation_v1", 2023)
    built = runner.build_base_args(cli, dataset_cfg, model_label, params, HORIZON, "matched_microbenchmark")
    built.seq_len = SEQUENCE_LENGTH
    built.label_len = SEQUENCE_LENGTH // 2
    built.batch_size = BATCH_SIZE
    experiment = Exp_Long_Term_Forecast(built)
    model = experiment.model.eval()
    x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, built.enc_in, device=experiment.device)
    decoder = torch.zeros(BATCH_SIZE, built.label_len + HORIZON, built.dec_in, device=experiment.device)
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x, None, decoder, None)
        torch.cuda.synchronize()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(REPEATS):
            started = time.perf_counter()
            for _ in range(ITERATIONS):
                model(x, None, decoder, None)
            torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - started) / ITERATIONS)
    row = {
        "model_id": model_label,
        "source_model_id": source_model_id(model_label),
        "model": display_name,
        "task": TASK,
        "horizon": HORIZON,
        "seq_len": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "warmup_iterations": WARMUP,
        "iterations_per_repeat": ITERATIONS,
        "repeats": REPEATS,
        "precision": "float32",
        "median_latency_ms": float(np.median(timings)),
        "p10_latency_ms": float(np.quantile(timings, 0.10)),
        "p90_latency_ms": float(np.quantile(timings, 0.90)),
        "throughput_samples_per_second": float(BATCH_SIZE * 1000.0 / np.median(timings)),
        "peak_allocated_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    del experiment, model, x, decoder
    torch.cuda.empty_cache()
    return row


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the matched microbenchmark")
    config = load_config()
    dataset_cfg = runner.load_dataset_manifest(
        REPO_ROOT / config["base_protocol"]["training_manifest"], TASK, REPO_ROOT
    )
    r6, _ = load_v3_params(config, TASK)
    r6.update(config["candidates"]["RegimeGatedMamba"]["parameter_overrides"])
    r6.update(loss="MSE", horizon_weighting="balanced")
    rows = [
        profile(runtime_model_name("RegimeGatedMamba"), public_name("RegimeGatedMamba"), r6, dataset_cfg),
        profile("PatchTST", "PatchTST", selected_baseline_params("PatchTST"), dataset_cfg),
        profile("TimesNet", "TimesNet", selected_baseline_params("TimesNet"), dataset_cfg),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
