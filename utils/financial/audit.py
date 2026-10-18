#!/usr/bin/env python
"""Audit archived historical baseline artifacts without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.metrics import expected_origins
from utils.financial.common import atomic_write_json, load_protocol


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "reproducibility"
    / "financial_reference_evaluation_v1"
    / "protocol"
    / "financial_confirmatory_source.json"
)
DEFAULT_MANIFEST = REPO_ROOT / "data" / "financial" / "manifests" / "training_datasets.json"
DEFAULT_SUMMARY = REPO_ROOT / "backup" / "results" / "financial_confirmatory" / "final_runs.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "backup" / "results" / "financial_confirmatory" / "runs"
DEFAULT_OUTPUT = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "audit" / "legacy_baseline_audit.json"


def _setting_for(run: dict, horizon: int) -> str | None:
    settings = run.get("settings", {})
    return settings.get(str(horizon), settings.get(horizon))


def audit(protocol_path: Path, manifest_path: Path, summary_path: Path, run_root: Path) -> dict:
    """Return a detailed pass/fail record for every legacy model/task/seed/horizon cell."""
    protocol = load_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("protocol_version") != protocol["protocol_version"]:
        errors.append("summary_protocol_version_mismatch")
    hashes = {entry.get("protocol_sha256") for entry in manifest.values()}
    if len(hashes) != 1 or payload.get("protocol_sha256") not in hashes:
        errors.append("summary_protocol_hash_mismatch")
    if payload.get("horizons") != protocol["horizons"]:
        errors.append("summary_horizons_mismatch")
    if payload.get("seeds") != protocol["final_seeds"]:
        errors.append("summary_seeds_mismatch")

    expected_keys = {
        f"{task}|{model}|{seed}"
        for task in manifest
        for model in protocol["level_models"]
        for seed in protocol["final_seeds"]
    }
    runs = payload.get("runs", {})
    complete_keys = {key for key, record in runs.items() if record.get("status") == "complete"}
    missing_keys = sorted(expected_keys - complete_keys)
    extra_keys = sorted(complete_keys - expected_keys)
    if missing_keys:
        errors.append("missing_complete_run_keys")
    if extra_keys:
        errors.append("unexpected_complete_run_keys")

    origins_expected = expected_origins(manifest, protocol)
    rows = []
    for run_key in sorted(expected_keys):
        task, model, seed_text = run_key.split("|")
        seed = int(seed_text)
        run = runs.get(run_key, {})
        for horizon in protocol["horizons"]:
            problems: list[str] = []
            setting = _setting_for(run, int(horizon))
            if run.get("status") != "complete":
                problems.append("run_not_complete")
            if not setting:
                problems.append("missing_setting")
                directory = None
            else:
                directory = run_root / setting
                if "FIN_SMOKE" in setting.upper():
                    problems.append("smoke_setting")
            if directory is not None:
                required = ["pred.npy", "true.npy", "origin_dates.npy", "metrics.npy", "meta.json"]
                for filename in required:
                    if not (directory / filename).is_file():
                        problems.append(f"missing_{filename}")
                if not problems:
                    try:
                        prediction = np.load(directory / "pred.npy", mmap_mode="r")
                        truth = np.load(directory / "true.npy", mmap_mode="r")
                        dates = np.load(directory / "origin_dates.npy", allow_pickle=False).astype(str)
                        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                        if prediction.shape != truth.shape or prediction.ndim != 3:
                            problems.append("prediction_shape_mismatch")
                        elif prediction.shape[1] != int(horizon):
                            problems.append("horizon_shape_mismatch")
                        if prediction.shape[0] != len(dates):
                            problems.append("origin_count_mismatch")
                        if not np.array_equal(dates, origins_expected[(task, int(horizon))]):
                            problems.append("origin_dates_mismatch")
                        expected_metadata = {
                            "protocol_version": protocol["protocol_version"],
                            "protocol_sha256": manifest[task]["protocol_sha256"],
                            "dataset_sha256": manifest[task]["data_sha256"],
                            "model_label": model,
                            "prediction_horizon": int(horizon),
                            "evaluation_split": "test",
                            "seed": seed,
                            "checkpoint_retained": False,
                        }
                        for field, expected in expected_metadata.items():
                            if metadata.get(field) != expected:
                                problems.append(f"metadata_{field}")
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        problems.append(f"unreadable_artifact:{type(exc).__name__}")
            rows.append({
                "run_key": run_key,
                "task": task,
                "model_id": model,
                "seed": seed,
                "horizon": int(horizon),
                "setting": setting or "",
                "passed": not problems,
                "problems": ";".join(problems),
            })

    failed_rows = [row for row in rows if not row["passed"]]
    return {
        "audit_version": "financial_reference_evaluation_v1",
        "read_only": True,
        "legacy_protocol_version": protocol["protocol_version"],
        "legacy_protocol_sha256": next(iter(hashes)) if len(hashes) == 1 else "",
        "expected_run_bundles": len(expected_keys),
        "complete_run_bundles": len(complete_keys),
        "expected_horizon_cells": len(expected_keys) * len(protocol["horizons"]),
        "passed_horizon_cells": len(rows) - len(failed_rows),
        "failed_horizon_cells": len(failed_rows),
        "missing_run_keys": missing_keys,
        "extra_run_keys": extra_keys,
        "summary_errors": errors,
        "compatible": not errors and not failed_rows,
        "rows": rows,
    }


def write_report(report: dict, output: Path) -> None:
    atomic_write_json(output, report)
    with (output.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(report["rows"][0]) if report["rows"] else [],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(report["rows"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.protocol, args.manifest, args.summary, args.run_root)
    write_report(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))
    if not report["compatible"]:
        raise SystemExit("Legacy baseline audit failed; do not use these records in publication tables.")


if __name__ == "__main__":
    main()
