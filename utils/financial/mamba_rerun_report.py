#!/usr/bin/env python
"""Rebuild metric tables from retained financial Mamba rerun artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.metrics import metrics, target_scales


CONFIG_PATH = REPO_ROOT / "configs" / "financial_mamba_rerun_v1.json"
OUTPUT_ROOT = REPO_ROOT / "results" / "financial_mamba_rerun_v1"
SUMMARY_NAME = "campaign_summary.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected(values: list[str], available: list[str], label: str) -> list[str]:
    if values == ["ALL"]:
        return list(available)
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; available={available}")
    return list(values)


def load_rows(config: dict, output_root: Path, tasks: list[str], models: list[str], horizons: list[int], seeds: list[int]) -> tuple[list[dict], list[str]]:
    summary_path = output_root / SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing campaign summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") != config["protocol_version"]:
        raise ValueError("Campaign summary protocol does not match report configuration")
    manifest = json.loads((REPO_ROOT / config["base_protocol"]["training_manifest"]).read_text(encoding="utf-8"))
    protocol = json.loads((REPO_ROOT / config["base_protocol"]["path"]).read_text(encoding="utf-8"))
    scales = target_scales(manifest, protocol)
    rows, problems = [], []
    for task in tasks:
        for model_id in models:
            for seed in seeds:
                key = f"{task}|{model_id}|{seed}"
                record = summary.get("runs", {}).get(key, {})
                if record.get("status") != "complete":
                    problems.append(f"missing_run:{key}")
                    continue
                for horizon in horizons:
                    setting = record.get("settings", {}).get(str(horizon))
                    directory = output_root / "runs" / str(setting)
                    required = [directory / name for name in ("pred.npy", "true.npy", "metrics.npy", "origin_dates.npy", "forecast_dates.npy", "meta.json")]
                    if not setting or any(not path.is_file() for path in required):
                        problems.append(f"missing_artifact:{key}:h{horizon}")
                        continue
                    prediction = np.load(directory / "pred.npy", mmap_mode="r")
                    truth = np.load(directory / "true.npy", mmap_mode="r")
                    forecast_dates = np.load(directory / "forecast_dates.npy", allow_pickle=False)
                    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                    if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[1] != horizon:
                        problems.append(f"shape_mismatch:{key}:h{horizon}")
                        continue
                    if forecast_dates.shape != prediction.shape[:2]:
                        problems.append(f"forecast_dates_mismatch:{key}:h{horizon}")
                        continue
                    expected_metadata = {
                        "campaign_protocol_version": config["protocol_version"],
                        "canonical_model_id": model_id,
                        "evaluation_split": "test",
                        "seed": seed,
                        "checkpoint_retained": True,
                        "forecast_data_retained": True,
                    }
                    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
                        problems.append(f"metadata_mismatch:{key}:h{horizon}")
                        continue
                    path_values = metrics(truth, prediction, scales[task])
                    terminal_values = metrics(truth[:, -1:, :], prediction[:, -1:, :], scales[task])
                    saved = np.load(directory / "metrics.npy")
                    if not np.isclose(float(saved[0]), path_values["mae"], rtol=1e-6, atol=1e-10):
                        problems.append(f"saved_mae_mismatch:{key}:h{horizon}")
                        continue
                    rows.append({
                        "dataset": task,
                        "model_id": model_id,
                        "source_model_id": record["source_model_id"],
                        "model": record["public_name"],
                        "seed": seed,
                        "horizon": horizon,
                        "origin_count": prediction.shape[0],
                        "runtime_seconds": float(metadata["runtime_seconds"]),
                        "peak_gpu_memory_mb": float(metadata["peak_gpu_memory_mb"]),
                        **{f"path_{name}": value for name, value in path_values.items()},
                        **{f"terminal_{name}": value for name, value in terminal_values.items()},
                        "setting": setting,
                    })
    return rows, problems


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    manifest = json.loads((REPO_ROOT / config["base_protocol"]["training_manifest"]).read_text(encoding="utf-8"))
    tasks = selected(args.tasks, sorted(manifest), "task IDs")
    models = selected(args.models, list(config["models"]), "model IDs")
    available_horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    horizons = [int(value) for value in selected([str(value) for value in (args.horizons or available_horizons)], [str(value) for value in available_horizons], "horizons")]
    available_seeds = [int(value) for value in config["base_protocol"]["seeds"]]
    seeds = [int(value) for value in selected([str(value) for value in args.seeds], [str(value) for value in available_seeds], "seeds")]
    rows, problems = load_rows(config, args.output_root, tasks, models, horizons, seeds)
    expected = len(tasks) * len(models) * len(horizons) * len(seeds)
    if problems and not args.allow_incomplete:
        raise RuntimeError(f"Incomplete campaign artifacts ({len(problems)} problems): {problems[:5]}")
    metrics_root = args.output_root / "metrics"
    write_csv(metrics_root / "metrics_by_seed.csv", rows)
    frame = pd.DataFrame(rows)
    metric_columns = [column for column in frame.columns if column.startswith(("path_", "terminal_", "runtime_", "peak_"))]
    aggregate = frame.groupby(["dataset", "model_id", "source_model_id", "model", "horizon"], as_index=False)[metric_columns].mean()
    aggregate.insert(5, "seed_count", frame.groupby(["dataset", "model_id", "source_model_id", "model", "horizon"]).size().to_numpy())
    aggregate.to_csv(metrics_root / "metrics_aggregate.csv", index=False)
    report = {
        "protocol_version": config["protocol_version"],
        "evidence_status": config["evidence_status"],
        "expected_rows": expected,
        "written_rows": len(rows),
        "problems": problems,
        "metrics_by_seed": str((metrics_root / "metrics_by_seed.csv").relative_to(REPO_ROOT)),
        "metrics_aggregate": str((metrics_root / "metrics_aggregate.csv").relative_to(REPO_ROOT)),
        "summary_sha256": sha256_path(args.output_root / SUMMARY_NAME),
    }
    atomic_write_json(metrics_root / "report_manifest.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="+", default=["ALL"])
    parser.add_argument("--models", nargs="+", default=["ALL"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025, 2026, 2027])
    parser.add_argument("--allow-incomplete", action="store_true")
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
        raise SystemExit(f"Mamba rerun report blocked: {exc}") from None
