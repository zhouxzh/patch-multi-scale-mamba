#!/usr/bin/env python
"""Build a conventional-error analysis from a verified S-only release.

The frozen release keeps RMSSE for protocol and audit reproducibility.  This module
provides the manuscript-facing reanalysis requested for the paper: original-scale
path MAE/RMSE and target-wise standard-deviation-normalized MAE/RMSE.  The
normalizing standard deviation is computed from the training levels only with
population convention (ddof=0), matching the StandardScaler used by the data
pipeline. No training one-step-naive forecast-error denominator is used.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.financial.common import REPO_ROOT, sha256_path


ANALYSIS_VERSION = "s_only_conventional_metrics_v1"
HORIZONS = (1, 5, 20, 60, 120, 252)
FAMILY_PREFIXES = OrderedDict(
    (
        ("Cross asset", "financial_cross_asset_"),
        ("Crypto", "financial_crypto_"),
        ("Foreign exchange", "financial_fx_"),
        ("Treasury yields", "financial_yields_"),
    )
)
TARGET_LABELS = {
    "financial_cross_asset_gld_s": "GLD",
    "financial_cross_asset_sp500_s": "S&P 500",
    "financial_cross_asset_vix_s": "VIX",
    "financial_cross_asset_wti_s": "WTI",
    "financial_crypto_bnbusdt_s": "BNBUSDT",
    "financial_crypto_btcusdt_s": "BTCUSDT",
    "financial_crypto_ethusdt_s": "ETHUSDT",
    "financial_crypto_xrpusdt_s": "XRPUSDT",
    "financial_fx_audusd_s": "AUDUSD",
    "financial_fx_cadusd_s": "CADUSD",
    "financial_fx_chfusd_s": "CHFUSD",
    "financial_fx_eurusd_s": "EURUSD",
    "financial_fx_gbpusd_s": "GBPUSD",
    "financial_fx_jpyusd_s": "JPYUSD",
    "financial_yields_dgs2_s": "DGS2",
    "financial_yields_dgs5_s": "DGS5",
    "financial_yields_dgs10_s": "DGS10",
    "financial_yields_dgs30_s": "DGS30",
}


def family_for(task: str) -> str:
    for family, prefix in FAMILY_PREFIXES.items():
        if task.startswith(prefix):
            return family
    raise ValueError(f"Unknown V2 task family: {task}")


def target_for(task: str) -> str:
    return TARGET_LABELS.get(task, task.removeprefix("financial_").removesuffix("_s"))


def _numeric(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = table.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def load_release(release: str) -> tuple[Path, pd.DataFrame, dict, dict]:
    root = REPO_ROOT / "reproducibility" / release
    status = json.loads((root / "release_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "verified" or status.get("problems"):
        raise RuntimeError(f"Analysis requires a verified release: {status}")
    aggregate = pd.read_csv(root / "metrics" / "metrics_aggregate.csv")
    manifest = json.loads((root / "training_datasets.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    required = {
        "dataset",
        "model_id",
        "model",
        "horizon",
        "seed_count",
        "path_mae_mean",
        "path_mse_mean",
        "path_rmse_mean",
    }
    missing = sorted(required - set(aggregate.columns))
    if missing:
        raise ValueError(f"Aggregate metrics are missing columns: {missing}")
    aggregate = _numeric(
        aggregate,
        [
            "horizon",
            "seed_count",
            "path_mae_mean",
            "path_mse_mean",
            "path_rmse_mean",
        ],
    )
    if len(manifest) != 18 or len(aggregate) != 18 * 14 * len(HORIZONS):
        raise ValueError(
            f"Unexpected V2 roster: targets={len(manifest)}, aggregate_rows={len(aggregate)}"
        )
    if set(aggregate["horizon"].astype(int).unique()) != set(HORIZONS):
        raise ValueError("Unexpected V2 horizon roster")
    if set(manifest) != set(aggregate["dataset"]):
        raise ValueError("Training manifest and aggregate dataset rosters differ")
    return root, aggregate, manifest, protocol


def training_scales(manifest: dict, protocol: dict) -> pd.DataFrame:
    """Return training-level SDs and source hashes for every target."""
    train_end = pd.Timestamp(protocol["splits"]["train_end"])
    rows = []
    for dataset, entry in sorted(manifest.items()):
        path = REPO_ROOT / entry["root_path"] / entry["data_path"]
        actual_hash = sha256_path(path)
        expected_hash = str(entry["data_sha256"]).lower()
        if actual_hash != expected_hash:
            raise ValueError(f"Training data hash mismatch for {dataset}")
        frame = pd.read_csv(path, usecols=["date", entry["target"]])
        dates = pd.to_datetime(frame["date"], errors="raise")
        values = pd.to_numeric(frame[entry["target"]], errors="raise").to_numpy(dtype=float)
        train_values = values[dates.to_numpy() <= train_end.to_datetime64()]
        if train_values.size < 2:
            raise ValueError(f"Too few training values for {dataset}")
        scale = float(np.std(train_values, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid training-level standard deviation for {dataset}")
        rows.append(
            {
                "dataset": dataset,
                "family": family_for(dataset),
                "target": target_for(dataset),
                "train_end": str(protocol["splits"]["train_end"]),
                "train_rows": int(train_values.size),
                "training_level_mean": float(np.mean(train_values)),
                "training_level_std_ddof0": scale,
                "data_sha256": actual_hash,
            }
        )
    return pd.DataFrame(rows)


def add_metrics(aggregate: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    scale_lookup = scales.set_index("dataset")["training_level_std_ddof0"]
    retained_columns = [
        "dataset",
        "model_id",
        "model",
        "source_model_id",
        "horizon",
        "seed_count",
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "path_mae_mean",
        "path_mae_std",
        "path_mse_mean",
        "path_mse_std",
        "path_rmse_mean",
        "path_rmse_std",
    ]
    missing = sorted(set(retained_columns) - set(aggregate.columns))
    if missing:
        raise ValueError(f"Aggregate metrics are missing conventional-analysis columns: {missing}")
    result = aggregate[retained_columns].copy()
    result["family"] = result["dataset"].map(family_for)
    result["target"] = result["dataset"].map(target_for)
    result["horizon"] = result["horizon"].astype(int)
    result["training_level_std_ddof0"] = result["dataset"].map(scale_lookup)
    if result["training_level_std_ddof0"].isna().any():
        raise ValueError("Missing training scale for at least one dataset")
    result["smae"] = result["path_mae_mean"] / result["training_level_std_ddof0"]
    result["smse"] = result["path_mse_mean"] / result["training_level_std_ddof0"].pow(2)
    result["srmse"] = result["path_rmse_mean"] / result["training_level_std_ddof0"]
    return result.sort_values(["dataset", "horizon", "model_id"]).reset_index(drop=True)


def summary(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = (
        table.groupby(group_columns, as_index=False)
        .agg(
            cells=("model_id", "size"),
            mean_smae=("smae", "mean"),
            median_smae=("smae", "median"),
            mean_srmse=("srmse", "mean"),
            median_srmse=("srmse", "median"),
        )
    )
    strata = [column for column in group_columns if column not in {"model_id", "model"}]
    return result.sort_values([*strata, "mean_srmse", "model_id"]).reset_index(drop=True)


def gate_contrast(table: pd.DataFrame) -> pd.DataFrame:
    columns = ["dataset", "family", "target", "horizon", "smae", "srmse"]
    gated = table[table["model_id"] == "RegimeGatedMamba"][columns].rename(
        columns={"smae": "gated_smae", "srmse": "gated_srmse"}
    )
    ungated = table[table["model_id"] == "UngatedRegimeLinearMamba"][columns].rename(
        columns={"smae": "ungated_smae", "srmse": "ungated_srmse"}
    )
    result = gated.merge(ungated, on=["dataset", "family", "target", "horizon"], how="inner")
    result["gated_minus_ungated_smae"] = result["gated_smae"] - result["ungated_smae"]
    result["gated_minus_ungated_srmse"] = result["gated_srmse"] - result["ungated_srmse"]
    result["gated_srmse_ratio"] = result["gated_srmse"] / result["ungated_srmse"]
    return result.sort_values(["dataset", "horizon"]).reset_index(drop=True)


def rounded(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    numeric = result.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].round(8)
    return result


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded(table).to_csv(path, index=False)


def manifest_path(path: Path) -> str:
    """Use stable repo-relative paths, while supporting isolated test outputs."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build(release: str, output_dir: Path) -> dict:
    if release != "financial_s_only_v2":
        raise ValueError("Conventional metrics are currently defined for financial_s_only_v2")
    root, aggregate, manifest, protocol = load_release(release)
    scales = training_scales(manifest, protocol)
    table = add_metrics(aggregate, scales)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, frame in (
        ("training_scales.csv", scales),
        ("cell_metrics.csv", table),
        ("model_summary.csv", summary(table, ["model_id", "model"])),
        ("family_summary.csv", summary(table, ["family", "model_id", "model"])),
        ("horizon_summary.csv", summary(table, ["horizon", "model_id", "model"])),
        ("gate_contrast.csv", gate_contrast(table)),
    ):
        path = output_dir / name
        write_table(frame, path)
        outputs.append(path)
    summary_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "release": release,
        "release_status": "verified",
        "retrospective_window": ["2023-01-01", "2026-06-30"],
        "cells": int(table[["dataset", "horizon"]].drop_duplicates().shape[0]),
        "methods": int(table["model_id"].nunique()),
        "primary_metrics": ["path_mae", "path_rmse"],
        "cross_target_metrics": ["smae", "srmse"],
        "normalization": {
            "definition": "training-level standard deviation",
            "formula": "sMAE = path MAE / SD_train; sRMSE = path RMSE / SD_train",
            "ddof": 0,
            "fit_boundary": "train_end inclusive (2020-12-31)",
            "model_error_denominator": False,
        },
        "rmsse_status": "retained only in the frozen release for audit; not used by this analysis",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(summary_path)
    source_paths = [
        root / "release_status.json",
        root / "protocol.json",
        root / "training_datasets.json",
        root / "metrics" / "metrics_aggregate.csv",
    ]
    manifest_payload = {
        "analysis_version": ANALYSIS_VERSION,
        "release": release,
        "source_files": [
            {"path": manifest_path(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "output_files": [
            {"path": manifest_path(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
        "transformations": [
            "Original-scale path MAE, MSE, and RMSE are taken from the verified aggregate table.",
            "Training-level standard deviations use each target's verified CSV and train_end inclusive.",
            "Standard deviations use ddof=0 to match sklearn StandardScaler.",
            "No RMSSE, training one-step-naive forecast-error denominator, cell rank, or winner count is used by this analysis.",
            "Gate contrasts are descriptive differences in standard-deviation-normalized MAE/RMSE.",
        ],
    }
    manifest_file = output_dir / "analysis_manifest.json"
    manifest_file.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(manifest_file)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "release": release,
        "output_dir": manifest_path(output_dir),
        "outputs": [manifest_path(path) for path in outputs],
        "status": "complete",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=("financial_s_only_v2",), default="financial_s_only_v2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "analysis" / "financial_s_only_v2_conventional_metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.release, args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
