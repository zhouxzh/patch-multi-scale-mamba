#!/usr/bin/env python
"""Analyze the local V2 supplemental gate-ablation release and render its figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import atomic_write_json, sha256_path


RELEASE = "financial_s_only_v2_supplement"
MODELS = ("RegimeGatedMamba", "UngatedRegimeLinearMamba")
MODEL_LABELS = {
    "RegimeGatedMamba": "Regime-Gated Mamba",
    "UngatedRegimeLinearMamba": "Ungated Regime-Linear Mamba",
}
TARGET_LABELS = {
    "financial_cross_asset_gld_s": "GLD", "financial_cross_asset_sp500_s": "S&P 500",
    "financial_cross_asset_vix_s": "VIX", "financial_cross_asset_wti_s": "WTI",
    "financial_crypto_bnbusdt_s": "BNBUSDT", "financial_crypto_btcusdt_s": "BTCUSDT",
    "financial_crypto_ethusdt_s": "ETHUSDT", "financial_crypto_xrpusdt_s": "XRPUSDT",
    "financial_fx_audusd_s": "AUDUSD", "financial_fx_cadusd_s": "CADUSD",
    "financial_fx_chfusd_s": "CHFUSD", "financial_fx_eurusd_s": "EURUSD",
    "financial_fx_gbpusd_s": "GBPUSD", "financial_fx_jpyusd_s": "JPYUSD",
    "financial_yields_dgs2_s": "DGS2", "financial_yields_dgs5_s": "DGS5",
    "financial_yields_dgs10_s": "DGS10", "financial_yields_dgs30_s": "DGS30",
}


def family_for(task: str) -> str:
    if task.startswith("financial_cross_asset_"):
        return "Cross asset"
    if task.startswith("financial_crypto_"):
        return "Crypto"
    if task.startswith("financial_fx_"):
        return "Foreign exchange"
    if task.startswith("financial_yields_"):
        return "Treasury yields"
    raise ValueError(f"Unknown V2 supplement task family: {task}")


def load_release(release: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    root = REPO_ROOT / "reproducibility" / release
    status = json.loads((root / "release_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "verified" or status.get("problems"):
        raise RuntimeError(f"V2 supplement analysis requires a verified release: {status}")
    seed = pd.read_csv(root / "metrics" / "metrics_by_seed.csv")
    aggregate = pd.read_csv(root / "metrics" / "metrics_aggregate.csv")
    bootstrap = pd.read_csv(root / "metrics" / "paired_block_bootstrap.csv")
    expected_seed = 18 * 2 * 5 * 3
    if len(seed) != expected_seed or set(seed["model_id"]) != set(MODELS):
        raise ValueError(f"V2 supplement seed table is not the planned 540-cell matrix: {len(seed)} rows")
    if len(aggregate) != 18 * 2 * 3:
        raise ValueError(f"Unexpected V2 supplement aggregate row count: {len(aggregate)}")
    if set(bootstrap["reference_model_id"]) != {"UngatedRegimeLinearMamba"}:
        raise ValueError("V2 supplement bootstrap table must compare the gate against its ungated ablation")
    return root, seed, aggregate, bootstrap, status


def _round(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    for column in result.select_dtypes(include=[np.number]).columns:
        result[column] = result[column].round(8)
    return result


def build_tables(seed: pd.DataFrame, aggregate: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    seed = seed.copy()
    aggregate = aggregate.copy()
    for table in (seed, aggregate):
        table["family"] = table["dataset"].map(family_for)
        table["target"] = table["dataset"].map(lambda value: TARGET_LABELS.get(value, value))
    model_summary = (
        seed.groupby(["model_id", "model"], as_index=False)
        .agg(cells=("path_rmsse", "size"), mean_path_mae=("path_mae", "mean"),
             mean_path_rmse=("path_rmse", "mean"), mean_path_rmsse=("path_rmsse", "mean"),
             mean_runtime_seconds=("runtime_seconds", "mean"))
    )
    model_summary["paired_rank"] = model_summary["mean_path_rmsse"].rank(method="min")
    horizon_summary = (
        seed.groupby(["horizon", "model_id", "model"], as_index=False)
        .agg(cells=("path_rmsse", "size"), mean_path_mae=("path_mae", "mean"),
             mean_path_rmse=("path_rmse", "mean"), mean_path_rmsse=("path_rmsse", "mean"),
             mean_runtime_seconds=("runtime_seconds", "mean"))
    )
    family_summary = (
        seed.groupby(["family", "model_id", "model"], as_index=False)
        .agg(cells=("path_rmsse", "size"), mean_path_mae=("path_mae", "mean"),
             mean_path_rmse=("path_rmse", "mean"), mean_path_rmsse=("path_rmsse", "mean"),
             mean_runtime_seconds=("runtime_seconds", "mean"))
    )
    cell = aggregate.pivot_table(index=["dataset", "target", "family", "horizon"], columns="model_id",
                                  values=["path_mae_mean", "path_rmse_mean", "path_rmsse_mean", "runtime_seconds_mean"])
    cell.columns = [f"{metric.removesuffix('_mean')}_{model}" for metric, model in cell.columns]
    cell = cell.reset_index()
    for metric in ("path_mae", "path_rmse", "path_rmsse", "runtime_seconds"):
        cell[f"gated_minus_ungated_{metric}"] = cell[f"{metric}_RegimeGatedMamba"] - cell[f"{metric}_UngatedRegimeLinearMamba"]
    cell["gate_favors_rmsse"] = cell["gated_minus_ungated_path_rmsse"] < 0
    seed_stability = (
        seed.groupby(["seed", "horizon", "model_id", "model"], as_index=False)
        .agg(target_count=("dataset", "nunique"), mean_path_mae=("path_mae", "mean"),
             mean_path_rmse=("path_rmse", "mean"), mean_path_rmsse=("path_rmsse", "mean"),
             mean_runtime_seconds=("runtime_seconds", "mean"))
    )
    return {
        "model_summary": _round(model_summary),
        "horizon_summary": _round(horizon_summary),
        "family_summary": _round(family_summary),
        "gate_cell_metrics": _round(cell),
        "seed_stability": _round(seed_stability),
        "bootstrap": _round(bootstrap),
    }


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    output = []
    for suffix, dpi in (("pdf", None), ("png", 400)):
        path = directory / f"financial_s_only_v2_supplement_{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        output.append(path)
    plt.close(fig)
    return output


def render_figures(tables: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    cell = tables["gate_cell_metrics"]
    pivot = cell.pivot(index="target", columns="horizon", values="gated_minus_ungated_path_rmsse")
    pivot = pivot.reindex(sorted(pivot.index), axis=0).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    maximum = float(np.nanmax(np.abs(pivot.to_numpy()))) or 1.0
    image = ax.imshow(pivot.to_numpy(), cmap="coolwarm_r", vmin=-maximum, vmax=maximum, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Target")
    ax.set_title("V2 supplemental gate effect: gated minus ungated path-RMSSE", loc="left", weight="bold")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Negative values favor Regime-Gated Mamba")
    outputs = save_figure(fig, directory, "gate_delta_heatmap")

    bootstrap = tables["bootstrap"].sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    x = np.arange(len(bootstrap))
    means = bootstrap["mean_path_rmsse_difference"].to_numpy()
    lows = means - bootstrap["bootstrap_ci_low"].to_numpy()
    highs = bootstrap["bootstrap_ci_high"].to_numpy() - means
    ax.errorbar(x, means, yerr=[lows, highs], fmt="o", color="#111111", capsize=4, linewidth=1.5)
    ax.axhline(0.0, color="#888888", linewidth=1)
    ax.set_xticks(x, [f"h={int(value)}" for value in bootstrap["horizon"]])
    ax.set_ylabel("Gated minus ungated path-RMSSE")
    ax.set_title("Paired bootstrap intervals for the V2 supplement\n(negative favors the gate)", loc="left", weight="bold")
    outputs.extend(save_figure(fig, directory, "gate_delta_bootstrap"))

    summary = tables["model_summary"].sort_values("mean_runtime_seconds")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    bars = ax.bar([MODEL_LABELS[m] for m in summary["model_id"]], summary["mean_runtime_seconds"], color=["#111111", "#888888"])
    ax.set_ylabel("Mean runtime per retained horizon run (s)")
    ax.set_title("Runtime and accuracy context for the V2 supplemental gate pair", loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=15)
    for bar, rank in zip(bars, summary.set_index("model_id").loc[summary["model_id"], "paired_rank"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"paired rank {rank:.0f}", ha="center", va="bottom", fontsize=8)
    outputs.extend(save_figure(fig, directory, "runtime_gate_pair"))
    return outputs


def _manifest_entry(path: Path) -> dict:
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_path(path), "bytes": path.stat().st_size}


def run(args: argparse.Namespace) -> dict:
    root, seed, aggregate, bootstrap, status = load_release(args.release)
    tables = build_tables(seed, aggregate, bootstrap)
    args.analysis_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, table in tables.items():
        if name == "bootstrap":
            continue
        path = args.analysis_output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs.append(path)
    bootstrap_path = args.analysis_output_dir / "paired_block_bootstrap.csv"
    tables["bootstrap"].to_csv(bootstrap_path, index=False)
    outputs.append(bootstrap_path)
    figures = render_figures(tables, args.figure_output_dir)
    summary = {
        "release": args.release,
        "evidence_status": "retrospective_supplement",
        "design": {"targets": 18, "models": 2, "seeds": 5, "horizons": [60, 120, 252], "learned_cells": 540},
        "reference_release": "financial_s_only_v2",
        "interpretation": "Supplementary paired gate-ablation evidence; the V2 baseline rankings remain the publication comparison authority.",
        "models": tables["model_summary"].to_dict(orient="records"),
    }
    summary_path = args.analysis_output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    outputs.append(summary_path)
    manifest = {
        "release": args.release,
        "analysis_version": "v2_supplement_gate_analysis_v1",
        "evidence_status": "retrospective_supplement",
        "source_files": [_manifest_entry(root / "metrics" / name) for name in ("metrics_by_seed.csv", "metrics_aggregate.csv", "paired_block_bootstrap.csv")],
        "output_files": [_manifest_entry(path) for path in [*outputs, *figures]],
    }
    manifest_path = args.analysis_output_dir / "analysis_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {"status": "complete", "release_status": status["status"], "analysis_dir": str(args.analysis_output_dir), "figure_count": len(figures), "table_count": len(outputs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=RELEASE)
    parser.add_argument("--analysis-output-dir", type=Path, default=REPO_ROOT / "docs" / "analysis" / RELEASE)
    parser.add_argument("--figure-output-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
