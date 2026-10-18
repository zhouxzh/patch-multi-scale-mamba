#!/usr/bin/env python
"""Build detailed, evidence-bound summaries for a verified S-only release."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from utils.financial.common import sha256_path


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_VERSION = "s_only_result_analysis_v1"
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
REFERENCES = (
    "LastValue",
    "RandomWalkDrift",
    "ARIMA",
    "PatchTST",
    "TimesNet",
    "UngatedRegimeLinearMamba",
    "iTransformer",
)


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


def load_inputs(release: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = REPO_ROOT / "reproducibility" / release
    status_path = root / "release_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "verified" or status.get("problems"):
        raise RuntimeError(f"Analysis requires a verified release: {status}")
    aggregate = pd.read_csv(root / "metrics" / "metrics_aggregate.csv")
    learned = pd.read_csv(root / "metrics" / "metrics_by_seed.csv")
    deterministic = pd.read_csv(root / "metrics" / "deterministic_metrics.csv")
    required_aggregate = {
        "dataset",
        "model_id",
        "model",
        "horizon",
        "seed_count",
        "path_mae_mean",
        "path_rmse_mean",
        "path_rmsse_mean",
        "path_rmsse_rank",
    }
    required_learned = {"dataset", "model_id", "seed", "horizon", "path_rmsse"}
    required_deterministic = {"dataset", "model_id", "horizon", "path_mae", "path_rmse", "path_rmsse"}
    for name, table, required in (
        ("aggregate", aggregate, required_aggregate),
        ("learned", learned, required_learned),
        ("deterministic", deterministic, required_deterministic),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} metrics are missing columns: {missing}")
    aggregate = _numeric(aggregate, ["horizon", "seed_count", "path_mae_mean", "path_rmse_mean", "path_rmsse_mean", "path_rmsse_rank"])
    learned = learned[learned["seed"].astype(str) != "deterministic"].copy()
    learned = _numeric(learned, ["seed", "horizon", "path_mae", "path_rmse", "path_rmsse"])
    deterministic = _numeric(deterministic, ["horizon", "path_mae", "path_rmse", "path_rmsse"])
    validate_inputs(status, aggregate, learned, deterministic)
    return root, aggregate, learned, deterministic, status


def validate_inputs(status: dict, aggregate: pd.DataFrame, learned: pd.DataFrame, deterministic: pd.DataFrame) -> None:
    if status.get("complete_learned_horizon_rows") != 5940 or status.get("expected_learned_horizon_rows") != 5940:
        raise ValueError("Expected exactly 5,940 completed learned V2 rows")
    if status.get("complete_deterministic_rows") != 324 or status.get("expected_deterministic_rows") != 324:
        raise ValueError("Expected exactly 324 completed deterministic V2 rows")
    tasks = set(aggregate["dataset"].unique())
    horizons = set(aggregate["horizon"].astype(int).unique())
    models = set(aggregate["model_id"].unique())
    if len(tasks) != 18 or horizons != set(HORIZONS) or len(models) != 14:
        raise ValueError(f"Unexpected V2 roster: tasks={len(tasks)}, horizons={horizons}, models={len(models)}")
    if len(aggregate) != 18 * 14 * len(HORIZONS):
        raise ValueError(f"Unexpected aggregate row count: {len(aggregate)}")
    if len(learned) != 5940 or len(deterministic) != 324:
        raise ValueError(f"Unexpected source row counts: learned={len(learned)}, deterministic={len(deterministic)}")
    if not tasks.issubset(TARGET_LABELS):
        raise ValueError(f"Unknown target labels: {sorted(tasks - set(TARGET_LABELS))}")


def add_labels(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["family"] = result["dataset"].map(family_for)
    result["target"] = result["dataset"].map(target_for)
    result["horizon"] = result["horizon"].astype(int)
    return result


def model_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    table = add_labels(aggregate)
    summary = (
        table.groupby(["model_id", "model"], as_index=False)
        .agg(
            cells=("path_rmsse_rank", "size"),
            mean_path_rmsse_rank=("path_rmsse_rank", "mean"),
            rank1_cells=("path_rmsse_rank", lambda values: int((values == 1).sum())),
            top3_cells=("path_rmsse_rank", lambda values: int((values <= 3).sum())),
        )
    )
    runtime = (
        table[table["seed_count"] > 0]
        .groupby("model_id", as_index=False)["runtime_seconds_mean"]
        .mean()
        .rename(columns={"runtime_seconds_mean": "mean_runtime_seconds"})
    )
    summary = summary.merge(runtime, on="model_id", how="left")
    summary["mean_path_rmsse_rank"] = summary["mean_path_rmsse_rank"].round(6)
    summary["mean_runtime_seconds"] = summary["mean_runtime_seconds"].round(6)
    return summary.sort_values(["mean_path_rmsse_rank", "model_id"]).reset_index(drop=True)


def grouped_summary(aggregate: pd.DataFrame, group: str) -> pd.DataFrame:
    table = add_labels(aggregate)
    grouped = (
        table.groupby([group, "model_id", "model"], as_index=False)
        .agg(
            cells=("path_rmsse_rank", "size"),
            mean_path_rmsse_rank=("path_rmsse_rank", "mean"),
            rank1_cells=("path_rmsse_rank", lambda values: int((values == 1).sum())),
            top3_cells=("path_rmsse_rank", lambda values: int((values <= 3).sum())),
        )
    )
    grouped["mean_path_rmsse_rank"] = grouped["mean_path_rmsse_rank"].round(6)
    return grouped.sort_values([group, "mean_path_rmsse_rank", "model_id"]).reset_index(drop=True)


def horizon_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    result = grouped_summary(aggregate, "horizon")
    runtime = (
        add_labels(aggregate)[lambda table: table["seed_count"] > 0]
        .groupby(["horizon", "model_id"], as_index=False)["runtime_seconds_mean"]
        .mean()
        .rename(columns={"runtime_seconds_mean": "mean_runtime_seconds"})
    )
    result = result.merge(runtime, on=["horizon", "model_id"], how="left")
    return result


def _rank_for_seed(learned: pd.DataFrame, deterministic: pd.DataFrame, task: str, horizon: int, seed: int) -> pd.DataFrame:
    learned_rows = learned[(learned["dataset"] == task) & (learned["horizon"] == horizon) & (learned["seed"] == seed)][["model_id", "path_rmsse"]].copy()
    deterministic_rows = deterministic[(deterministic["dataset"] == task) & (deterministic["horizon"] == horizon)][["model_id", "path_rmsse"]].copy()
    values = pd.concat([learned_rows, deterministic_rows], ignore_index=True)
    values["rank"] = values["path_rmsse"].rank(method="min")
    return values


def regime_gated_cells(aggregate: pd.DataFrame, learned: pd.DataFrame, deterministic: pd.DataFrame) -> pd.DataFrame:
    table = add_labels(aggregate)
    proposed = table[table["model_id"] == "RegimeGatedMamba"].copy()
    lookup = table.set_index(["dataset", "horizon", "model_id"])
    rows: list[dict] = []
    for row in proposed.itertuples(index=False):
        key = (row.dataset, int(row.horizon))
        output = {
            "dataset": row.dataset,
            "family": row.family,
            "target": row.target,
            "horizon": int(row.horizon),
            "rgm_path_mae": float(row.path_mae_mean),
            "rgm_path_rmse": float(row.path_rmse_mean),
            "rgm_path_rmsse": float(row.path_rmsse_mean),
            "rgm_path_rmsse_rank": float(row.path_rmsse_rank),
        }
        for reference in REFERENCES:
            reference_row = lookup.loc[(row.dataset, int(row.horizon), reference)]
            value = float(reference_row["path_rmsse_mean"])
            output[f"{reference}_path_rmsse"] = value
            output[f"rgm_minus_{reference}_path_rmsse"] = output["rgm_path_rmsse"] - value
        last = lookup.loc[(row.dataset, int(row.horizon), "LastValue")]
        for metric, column in (("mae", "path_mae_mean"), ("rmse", "path_rmse_mean"), ("rmsse", "path_rmsse_mean")):
            output[f"rgm_over_lastvalue_{metric}_ratio"] = float(getattr(row, column) / float(last[column]))
        seed_values = learned[(learned["dataset"] == row.dataset) & (learned["model_id"] == "RegimeGatedMamba") & (learned["horizon"] == int(row.horizon))]["path_rmsse"]
        output["rgm_seed_count"] = int(seed_values.size)
        output["rgm_seed_path_rmsse_sd"] = float(seed_values.std(ddof=1))
        output["rgm_seed_path_rmsse_min"] = float(seed_values.min())
        output["rgm_seed_path_rmsse_max"] = float(seed_values.max())
        output["rgm_seed_path_rmsse_cv"] = float(seed_values.std(ddof=1) / seed_values.mean())
        seed_ranks = []
        for seed in sorted(learned["seed"].unique().astype(int)):
            ranked = _rank_for_seed(learned, deterministic, row.dataset, int(row.horizon), seed)
            proposed_rank = float(ranked.loc[ranked["model_id"] == "RegimeGatedMamba", "rank"].iloc[0])
            seed_ranks.append(proposed_rank)
        output["rgm_seed_rank_mean"] = float(np.mean(seed_ranks))
        output["rgm_seed_rank_sd"] = float(np.std(seed_ranks, ddof=1))
        output["rgm_seed_top3_fraction"] = float(np.mean(np.asarray(seed_ranks) <= 3))
        rows.append(output)
    result = pd.DataFrame(rows)
    numeric = [column for column in result.columns if column not in {"dataset", "family", "target"}]
    result[numeric] = result[numeric].round(8)
    return result.sort_values(["family", "target", "horizon"]).reset_index(drop=True)


def pairwise_summary(cells: pd.DataFrame) -> list[dict]:
    summaries = []
    for reference in ("LastValue", "RandomWalkDrift", "ARIMA", "UngatedRegimeLinearMamba", "PatchTST", "TimesNet", "iTransformer"):
        delta = cells[f"rgm_minus_{reference}_path_rmsse"]
        summaries.append(
            {
                "reference_model_id": reference,
                "cells": int(delta.size),
                "strict_wins": int((delta < 0).sum()),
                "strict_losses": int((delta > 0).sum()),
                "ties": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
                "mean_rgm_minus_reference_path_rmsse": round(float(delta.mean()), 8),
            }
        )
    return summaries


def target_summary(cells: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cells.groupby(["dataset", "family", "target"], as_index=False)
        .agg(
            horizons=("horizon", "size"),
            mean_path_mae=("rgm_path_mae", "mean"),
            mean_path_rmse=("rgm_path_rmse", "mean"),
            mean_path_rmsse=("rgm_path_rmsse", "mean"),
            mean_rank=("rgm_path_rmsse_rank", "mean"),
            rank1_cells=("rgm_path_rmsse_rank", lambda values: int((values == 1).sum())),
            top3_cells=("rgm_path_rmsse_rank", lambda values: int((values <= 3).sum())),
            mean_lastvalue_rmsse_ratio=("rgm_over_lastvalue_rmsse_ratio", "mean"),
            mean_lastvalue_mae_ratio=("rgm_over_lastvalue_mae_ratio", "mean"),
            mean_lastvalue_rmse_ratio=("rgm_over_lastvalue_rmse_ratio", "mean"),
        )
    )
    numeric = [column for column in summary.columns if column not in {"dataset", "family", "target"}]
    summary[numeric] = summary[numeric].round(8)
    return summary.sort_values(["family", "mean_rank", "target"]).reset_index(drop=True)


def build_summary(status: dict, aggregate: pd.DataFrame, cells: pd.DataFrame, bootstrap: pd.DataFrame) -> dict:
    rgm_horizon = (
        cells.groupby("horizon", as_index=False)
        .agg(mean_rank=("rgm_path_rmsse_rank", "mean"), rank1_cells=("rgm_path_rmsse_rank", lambda values: int((values == 1).sum())), top3_cells=("rgm_path_rmsse_rank", lambda values: int((values <= 3).sum())))
        .sort_values("horizon")
    )
    rgm_family = (
        cells.groupby("family", as_index=False)
        .agg(mean_rank=("rgm_path_rmsse_rank", "mean"), rank1_cells=("rgm_path_rmsse_rank", lambda values: int((values == 1).sum())), top3_cells=("rgm_path_rmsse_rank", lambda values: int((values <= 3).sum())))
    )
    gate = bootstrap[(bootstrap["model_id"] == "RegimeGatedMamba") & (bootstrap["reference_model_id"] == "UngatedRegimeLinearMamba")].sort_values("horizon")
    runtime = aggregate[aggregate["model_id"].isin(["RegimeGatedMamba", "PatchTST", "TimesNet", "UngatedRegimeLinearMamba", "iTransformer"]) & (aggregate["seed_count"] > 0)].groupby("model_id")["runtime_seconds_mean"].mean()
    replay_validation = REPO_ROOT / "results" / "financial_s_only_retrospective_v2" / "replay" / "replay_validation.json"
    replay_audit = "not_completed_for_v2"
    if replay_validation.exists():
        try:
            replay_status = json.loads(replay_validation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            replay_status = {}
        if (
            replay_status.get("status") == "verified"
            and replay_status.get("expected_entries") == 5940
            and replay_status.get("validated_entries") == 5940
            and not replay_status.get("problems")
        ):
            replay_audit = "verified"
    return {
        "analysis_version": ANALYSIS_VERSION,
        "release": "financial_s_only_v2",
        "release_status": status["status"],
        "completed_rows": {"learned": int(status["complete_learned_horizon_rows"]), "deterministic": int(status["complete_deterministic_rows"]), "cells": int(len(cells))},
        "retrospective_window": ["2023-01-01", "2026-06-30"],
        "primary_metric": "path_rmsse",
        "rgm_overall": {"mean_path_rmsse_rank": round(float(cells["rgm_path_rmsse_rank"].mean()), 6), "rank1_cells": int((cells["rgm_path_rmsse_rank"] == 1).sum()), "top3_cells": int((cells["rgm_path_rmsse_rank"] <= 3).sum())},
        "rgm_by_horizon": [{"horizon": int(row.horizon), "mean_rank": round(float(row.mean_rank), 6), "rank1_cells": int(row.rank1_cells), "top3_cells": int(row.top3_cells)} for row in rgm_horizon.itertuples()],
        "rgm_by_family": [{"family": row.family, "mean_rank": round(float(row.mean_rank), 6), "rank1_cells": int(row.rank1_cells), "top3_cells": int(row.top3_cells)} for row in rgm_family.itertuples()],
        "pairwise_path_rmsse": pairwise_summary(cells),
        "gate_bootstrap": gate[["horizon", "mean_path_rmsse_difference", "bootstrap_ci_low", "bootstrap_ci_high", "p_value_holm"]].round(8).to_dict("records"),
        "mean_runtime_seconds": {str(model): round(float(value), 6) for model, value in runtime.items()},
        "replay_audit": replay_audit,
        "claim_boundary": "retrospective target-only metrics, ablation, and fixed-workload runtime only",
    }


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def build(release: str, output_dir: Path) -> dict:
    if release != "financial_s_only_v2":
        raise ValueError("Detailed result analysis is currently defined for financial_s_only_v2")
    root, aggregate, learned, deterministic, status = load_inputs(release)
    aggregate = add_labels(aggregate)
    cells = regime_gated_cells(aggregate, learned, deterministic)
    bootstrap = pd.read_csv(root / "metrics" / "paired_block_bootstrap.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    tables = {
        "model_summary.csv": model_summary(aggregate),
        "horizon_summary.csv": horizon_summary(aggregate),
        "family_summary.csv": grouped_summary(aggregate, "family"),
        "target_summary.csv": target_summary(cells),
        "regime_gated_cell_metrics.csv": cells,
        "seed_stability.csv": cells[["dataset", "family", "target", "horizon", "rgm_seed_count", "rgm_seed_path_rmsse_sd", "rgm_seed_path_rmsse_min", "rgm_seed_path_rmsse_max", "rgm_seed_path_rmsse_cv", "rgm_seed_rank_mean", "rgm_seed_rank_sd", "rgm_seed_top3_fraction"]],
    }
    for name, table in tables.items():
        path = output_dir / name
        write_table(table, path)
        output_paths.append(path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(build_summary(status, aggregate, cells, bootstrap), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths.append(summary_path)
    source_paths = [
        root / "release_status.json",
        root / "metrics" / "metrics_by_seed.csv",
        root / "metrics" / "metrics_aggregate.csv",
        root / "metrics" / "deterministic_metrics.csv",
        root / "metrics" / "paired_block_bootstrap.csv",
    ]
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "release": release,
        "source_files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_path(path), "bytes": path.stat().st_size} for path in source_paths],
        "output_files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_path(path), "bytes": path.stat().st_size} for path in output_paths],
        "transformations": [
            "Ranks are computed within each target-horizon cell with lower path RMSSE better.",
            "Raw MAE and RMSE are retained per target and are not pooled across target scales.",
            "Cross-target metric summaries use ratios to LastValue; signed RMSSE deltas use named references.",
            "Seed stability uses the five learned seeds and deterministic references at the same target and horizon.",
        ],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths.append(manifest_path)
    return {"release": release, "output_dir": str(output_dir.relative_to(REPO_ROOT)), "outputs": [str(path.relative_to(REPO_ROOT)) for path in output_paths], "status": "complete"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=("financial_s_only_v2",), default="financial_s_only_v2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "analysis" / "financial_s_only_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.release, args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
