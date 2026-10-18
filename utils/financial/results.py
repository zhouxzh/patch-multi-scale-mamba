#!/usr/bin/env python
"""Build final evaluation tables from audited financial reference records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.metrics import expected_origins, metrics, target_scales
from utils.financial.common import atomic_write_json
from utils.financial.audit import DEFAULT_OUTPUT as AUDIT_OUTPUT
from utils.financial.audit import audit
from utils.financial.registry import canonical_model_id, public_name, source_model_id


CONFIG_PATH = REPO_ROOT / "configs" / "financial_reference_evaluation_v1.json"
LEGACY_METRICS = REPO_ROOT / "backup" / "results" / "financial_confirmatory" / "analysis" / "neural_metrics_by_seed.csv"
# Completed candidate run artifacts keep their historical local directory. New runs use
# ``results/financial_reference_evaluation_v1`` through the financial reference runner.
CANDIDATE_ROOT = REPO_ROOT / "results" / "financial_reference_evaluation_v1"
DEFAULT_OUTPUT = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "metrics"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_rows(config: dict, output_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Recompute candidate metrics from row-level artifacts, never summary scalars."""
    base = config["base_protocol"]
    manifest_path = REPO_ROOT / base["training_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads((REPO_ROOT / base["path"]).read_text(encoding="utf-8"))
    summary_path = output_root / "final_runs.json"
    if not summary_path.is_file():
        return pd.DataFrame(), ["candidate_summary_missing"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") != config["protocol_version"]:
        return pd.DataFrame(), ["candidate_protocol_version_mismatch"]
    scales = target_scales(manifest, protocol)
    origins = expected_origins(manifest, protocol)
    expected = {
        f"{task}|{candidate}|{seed}"
        for task in manifest
        for candidate in config["candidates"]
        for seed in base["seeds"]
    }
    rows, problems = [], []
    for key in sorted(expected):
        task, candidate_id, seed_text = key.split("|")
        seed = int(seed_text)
        record = summary.get("runs", {}).get(key, {})
        if record.get("status") != "complete":
            problems.append(f"missing_run:{key}")
            continue
        settings = record.get("settings", {})
        for horizon in base["horizons"]:
            setting = settings.get(str(horizon), settings.get(horizon))
            directory = output_root / "runs" / str(setting)
            required = [directory / name for name in ("pred.npy", "true.npy", "origin_dates.npy", "meta.json")]
            if not setting or any(not path.is_file() for path in required):
                problems.append(f"missing_artifact:{key}:h{horizon}")
                continue
            prediction = np.load(directory / "pred.npy", mmap_mode="r")
            truth = np.load(directory / "true.npy", mmap_mode="r")
            dates = np.load(directory / "origin_dates.npy", allow_pickle=False).astype(str)
            metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[1] != int(horizon):
                problems.append(f"shape_mismatch:{key}:h{horizon}")
                continue
            if not np.array_equal(dates, origins[(task, int(horizon))]):
                problems.append(f"origin_mismatch:{key}:h{horizon}")
                continue
            required_metadata = {
                "publication_protocol_version": config["protocol_version"],
                "publication_model_id": candidate_id,
                "evaluation_status": "retrospective_protocol_matched",
                "evaluation_split": "test",
                "seed": seed,
            }
            if any(metadata.get(field) != value for field, value in required_metadata.items()):
                problems.append(f"metadata_mismatch:{key}:h{horizon}")
                continue
            path = metrics(truth, prediction, scales[task])
            terminal = metrics(truth[:, -1:, :], prediction[:, -1:, :], scales[task])
            runtime = float(metadata.get("runtime_seconds", float("nan")))
            memory = float(metadata.get("peak_gpu_memory_mb", float("nan")))
            if not math.isfinite(runtime) or runtime <= 0 or not math.isfinite(memory) or memory <= 0:
                problems.append(f"runtime_metadata_invalid:{key}:h{horizon}")
                continue
            rows.append({
                "dataset": task,
                "model_id": candidate_id,
                "source_model_id": candidate["source_model_id"],
                "model": public_name(candidate_id),
                "evidence_layer": "retrospective_protocol_matched",
                "seed": seed,
                "horizon": int(horizon),
                "origin_count": len(dates),
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": memory,
                **{f"path_{metric}": value for metric, value in path.items()},
                **{f"terminal_{metric}": value for metric, value in terminal.items()},
                "setting": str(setting),
            })
    return pd.DataFrame(rows), problems


def legacy_rows() -> pd.DataFrame:
    frame = pd.read_csv(LEGACY_METRICS)
    frame.insert(1, "source_model_id", frame["model"])
    frame.insert(2, "model_id", frame["source_model_id"].map(canonical_model_id))
    frame["model"] = frame["model_id"].map(public_name)
    frame.insert(4, "evidence_layer", "audited_legacy_baseline")
    return frame


def aggregate_rows(rows: pd.DataFrame, term_groups: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_map = {int(horizon): term for term, horizons in term_groups.items() for horizon in horizons}
    rows = rows.copy()
    rows["term"] = rows["horizon"].map(term_map)
    aggregate = (
        rows.groupby(["evidence_layer", "dataset", "model_id", "model", "horizon"], as_index=False)
        .agg(**{
            f"{metric}_mean": (metric, "mean")
            for metric in (
                "path_mae", "path_mse", "path_rmse", "path_rmsse",
                "terminal_mae", "terminal_mse", "terminal_rmse", "terminal_rmsse",
            )
        }, runtime_seconds_mean=("runtime_seconds", "mean"),
           peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"))
    )
    by_term = (
        rows.groupby(["evidence_layer", "dataset", "model_id", "model", "seed", "term"], as_index=False)
        .agg(path_rmsse=("path_rmsse", "mean"))
    )
    balanced = (
        by_term.groupby(["evidence_layer", "dataset", "model_id", "model", "seed"], as_index=False)
        .agg(balanced_path_rmsse=("path_rmsse", "mean"))
    )
    return aggregate, balanced


def paired_contrasts(rows: pd.DataFrame, candidate_ids: list[str]) -> pd.DataFrame:
    candidates = rows[rows["model_id"].isin(candidate_ids)].copy()
    references = rows[~rows["model_id"].isin(candidate_ids)].copy()
    results = []
    for candidate_id in candidate_ids:
        left = candidates[candidates.model_id == candidate_id]
        for reference_id in sorted(references.model_id.unique()):
            right = references[references.model_id == reference_id]
            merge = left.merge(right, on=["dataset", "seed", "horizon"], suffixes=("_candidate", "_reference"))
            if merge.empty:
                continue
            difference = merge.path_rmsse_candidate - merge.path_rmsse_reference
            results.append({
                "candidate_id": candidate_id,
                "candidate": public_name(candidate_id),
                "reference_id": reference_id,
                "reference": public_name(reference_id),
                "matched_cells": len(merge),
                "mean_path_rmsse_difference": float(difference.mean()),
                "candidate_win_fraction": float((difference < 0).mean()),
            })
    return pd.DataFrame(results)


def analyze(config_path: Path, candidate_root: Path, output: Path, require_complete: bool) -> dict:
    config = load_config(config_path)
    base = config["base_protocol"]
    audit_report = audit(
        REPO_ROOT / base["path"], REPO_ROOT / base["training_manifest"],
        REPO_ROOT / base["legacy_final_summary"], REPO_ROOT / base["legacy_run_root"],
    )
    atomic_write_json(AUDIT_OUTPUT, audit_report)
    if not audit_report["compatible"]:
        raise RuntimeError("Legacy audit failed; no publication table may be generated")
    legacy = legacy_rows()
    candidates, candidate_problems = candidate_rows(config, candidate_root)
    complete = not candidate_problems
    status = {
        "publication_protocol_version": config["protocol_version"],
        "legacy_baselines_audited": True,
        "candidate_matrix_complete": complete,
        "candidate_problems": candidate_problems,
        "candidate_row_count": int(len(candidates)),
        "legacy_row_count": int(len(legacy)),
        "claim_boundary": "Candidate comparisons are retrospective and protocol-matched; they are not an untouched test claim.",
    }
    output.mkdir(parents=True, exist_ok=True)
    legacy.to_csv(output / "audited_legacy_baseline_metrics_by_seed.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(output / "candidate_models_metrics_by_seed.csv", index=False)
    if require_complete and not complete:
        atomic_write_json(output / "release_status.json", status)
        raise RuntimeError("Final candidate matrix is incomplete")
    if complete:
        combined = pd.concat([legacy, candidates], ignore_index=True)
        combined.to_csv(output / "all_model_metrics_by_seed.csv", index=False)
        aggregate, balanced = aggregate_rows(combined, json.loads((REPO_ROOT / base["path"]).read_text())["term_groups"])
        aggregate.to_csv(output / "all_model_aggregate_metrics.csv", index=False)
        balanced.to_csv(output / "balanced_path_rmsse_by_task_seed.csv", index=False)
        paired_contrasts(combined, list(config["candidates"])).to_csv(
            output / "regime_gated_mamba_paired_contrasts.csv", index=False
        )
        status["combined_row_count"] = int(len(combined))
    atomic_write_json(output / "release_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(analyze(args.config, args.candidate_root, args.output, args.require_complete), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Publication analysis blocked: {exc}") from None
