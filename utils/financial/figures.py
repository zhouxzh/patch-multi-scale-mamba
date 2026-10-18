#!/usr/bin/env python
"""Render evidence-bound figures from a verified S-only financial release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_LABELS = {
    "financial_cross_asset_s": "Cross-asset (GLD)",
    "financial_crypto_s": "Crypto (BTCUSDT)",
    "financial_fx_s": "Foreign exchange (GBPUSD)",
    "financial_yields_s": "Treasury yields (DGS10)",
}
TASK_ABBREVIATIONS = {
    "financial_cross_asset_s": "Cross asset",
    "financial_crypto_s": "Crypto",
    "financial_fx_s": "FX",
    "financial_yields_s": "Yields",
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
V2_FAMILY_SPECS = (
    ("Cross asset", "financial_cross_asset_"),
    ("Crypto", "financial_crypto_"),
    ("Foreign exchange", "financial_fx_"),
    ("Treasury yields", "financial_yields_"),
)
MODEL_LABELS = {
    "RegimeGatedMamba": "Regime-Gated Mamba",
    "UngatedRegimeLinearMamba": "Ungated Regime-Linear Mamba",
    "MultiScalePatchMamba": "Multi-Scale Patch Mamba",
    "InvertedMamba": "Inverted Mamba",
    "DecompositionMamba": "Decomposition Mamba",
    "MultiScaleDecompositionMamba": "Multi-Scale Decomposition Mamba",
    "DLinear": "DLinear",
    "NLinear": "NLinear",
    "PatchTST": "PatchTST",
    "TimesNet": "TimesNet",
    "iTransformer": "iTransformer",
    "LastValue": "Last-value",
    "RandomWalkDrift": "Random-walk-with-drift",
    "ARIMA": "ARIMA",
}
MODEL_STYLES = {
    "RegimeGatedMamba": {"color": "#000000", "marker": "D", "linestyle": "-", "linewidth": 2.4, "zorder": 5},
    "UngatedRegimeLinearMamba": {"color": "#666666", "marker": "d", "linestyle": "--", "linewidth": 1.5},
    "MultiScalePatchMamba": {"color": "#0072B2", "marker": "o", "linestyle": "-", "linewidth": 1.2},
    "InvertedMamba": {"color": "#009E73", "marker": "s", "linestyle": "-", "linewidth": 1.2},
    "DecompositionMamba": {"color": "#56B4E9", "marker": "^", "linestyle": "-", "linewidth": 1.2},
    "MultiScaleDecompositionMamba": {"color": "#CC79A7", "marker": "v", "linestyle": "-", "linewidth": 1.2},
    "DLinear": {"color": "#E69F00", "marker": "P", "linestyle": "--", "linewidth": 1.2},
    "NLinear": {"color": "#D55E00", "marker": "X", "linestyle": "--", "linewidth": 1.2},
    "PatchTST": {"color": "#8C564B", "marker": "<", "linestyle": "-.", "linewidth": 1.2},
    "TimesNet": {"color": "#984EA3", "marker": ">", "linestyle": "-.", "linewidth": 1.2},
    "iTransformer": {"color": "#4D4D4D", "marker": "h", "linestyle": "-.", "linewidth": 1.2},
    "LastValue": {"color": "#1B9E77", "marker": "o", "linestyle": ":", "linewidth": 1.4},
    "RandomWalkDrift": {"color": "#7570B3", "marker": "s", "linestyle": ":", "linewidth": 1.4},
    "ARIMA": {"color": "#E7298A", "marker": "^", "linestyle": ":", "linewidth": 1.4},
}
FORECAST_MODEL_IDS = ("RegimeGatedMamba", "PatchTST", "TimesNet")
FORECAST_COLORS = {
    "RegimeGatedMamba": "#000000",
    "PatchTST": "#8C564B",
    "TimesNet": "#984EA3",
    "ARIMA": "#E7298A",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release(release: str) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    root = REPO_ROOT / "reproducibility" / release
    status_path = root / "release_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "verified":
        raise RuntimeError(f"Figures require a verified release, found {status.get('status')!r}")
    aggregate_path = root / "metrics" / "metrics_aggregate.csv"
    seed_path = root / "metrics" / "metrics_by_seed.csv"
    if not aggregate_path.is_file() or not seed_path.is_file():
        raise FileNotFoundError("Verified release is missing aggregate or seed-level metrics")
    return root, pd.read_csv(aggregate_path), pd.read_csv(seed_path), status


def model_order(table: pd.DataFrame) -> list[str]:
    ranks = table.groupby("model_id", as_index=True)["path_rmsse_rank"].mean().sort_values()
    return ranks.index.tolist()


def release_display_name(release: str) -> str:
    return {
        "financial_s_only_v1": "S-only v1",
        "financial_s_only_v2": "S-only v2",
    }[release]


def v2_family_tasks(table: pd.DataFrame) -> list[tuple[str, list[str]]]:
    available = set(table["dataset"].unique())
    return [
        (label, sorted(task for task in available if task.startswith(prefix)))
        for label, prefix in V2_FAMILY_SPECS
    ]


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, dpi in (("pdf", None), ("png", 450)):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=dpi, facecolor="white")
        if suffix == "png":
            with Image.open(path) as image:
                image.convert("RGB").save(path, format="PNG", dpi=image.info.get("dpi", (dpi, dpi)))
        paths.append(path)
    plt.close(figure)
    return paths


def plot_rank_heatmap(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    tasks = sorted(table["dataset"].unique())
    horizons = sorted(table["horizon"].unique())
    models = model_order(table)
    columns = [(task, horizon) for task in tasks for horizon in horizons]
    matrix = np.array(
        [
            [
                table.loc[(table["model_id"] == model) & (table["dataset"] == task) & (table["horizon"] == horizon), "path_rmsse_rank"].iloc[0]
                for task, horizon in columns
            ]
            for model in models
        ],
        dtype=float,
    )
    max_rank = int(np.nanmax(matrix))
    figure, axis = plt.subplots(figsize=(12.4, 6.35), layout="constrained")
    cmap = plt.colormaps["viridis_r"].resampled(max_rank)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(0.5, max_rank + 1.5), cmap.N),
        interpolation="nearest",
    )
    axis.set_xticks(
        np.arange(len(columns)),
        [f"{TASK_ABBREVIATIONS.get(task, task)}\n{horizon}" for task, horizon in columns],
        fontsize=6.5,
    )
    axis.set_yticks(np.arange(len(models)), [MODEL_LABELS.get(model, model) for model in models], fontsize=7)
    axis.set_xlabel("Forecast horizon (retained observations)")
    axis.set_ylabel("Model")
    axis.set_title("S-only v1 path-RMSSE rank by task and horizon", loc="left", fontsize=12, weight="bold", pad=27)
    axis.text(
        0,
        1.005,
        "Rank 1 is the lowest path RMSSE in a task-horizon cell; learned rows are five-seed means.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    for boundary in range(len(horizons), len(columns), len(horizons)):
        axis.axvline(boundary - 0.5, color="white", linewidth=2.0)
    for row_index, model in enumerate(models):
        for column_index, rank in enumerate(matrix[row_index]):
            text_color = "white" if rank >= max_rank * 0.65 else "black"
            axis.text(column_index, row_index, f"{int(rank)}", ha="center", va="center", fontsize=5.2, color=text_color)
        if model == "RegimeGatedMamba":
            axis.add_patch(plt.Rectangle((-0.5, row_index - 0.5), len(columns), 1, fill=False, edgecolor="#000000", linewidth=1.2))
    colorbar = figure.colorbar(image, ax=axis, pad=0.01, fraction=0.03)
    colorbar.set_label("Path-RMSSE rank (1 = best)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return save_figure(figure, output_dir, f"{release}_path_rmsse_rank_heatmap")


def plot_v2_rank_heatmap(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Show every model rank without compressing the 18-target release into one scale."""
    horizons = sorted(table["horizon"].unique())
    models = model_order(table)
    max_rank = int(table["path_rmsse_rank"].max())
    cmap = plt.colormaps["viridis_r"].resampled(max_rank)
    norm = BoundaryNorm(np.arange(0.5, max_rank + 1.5), cmap.N)
    figure = plt.figure(figsize=(14.25, 11.1), layout="constrained")
    main, note_band = figure.subfigures(2, 1, height_ratios=[12, 0.9])
    axes = list(main.subplots(2, 2).flat)
    image = None
    for axis, (family, tasks) in zip(axes, v2_family_tasks(table)):
        columns = [(task, horizon) for task in tasks for horizon in horizons]
        matrix = np.array(
            [
                [
                    table.loc[
                        (table["model_id"] == model)
                        & (table["dataset"] == task)
                        & (table["horizon"] == horizon),
                        "path_rmsse_rank",
                    ].iloc[0]
                    for task, horizon in columns
                ]
                for model in models
            ],
            dtype=float,
        )
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        axis.set_xticks(np.arange(len(columns)), [str(horizon) for _, horizon in columns], fontsize=4.5)
        target_axis = axis.secondary_xaxis("top")
        target_axis.set_xticks(
            [index * len(horizons) + (len(horizons) - 1) / 2 for index in range(len(tasks))],
            [TASK_ABBREVIATIONS.get(task, task) for task in tasks],
            fontsize=5.3,
        )
        target_axis.tick_params(axis="x", which="both", length=0, pad=1.5)
        axis.set_yticks(np.arange(len(models)), [MODEL_LABELS.get(model, model) for model in models], fontsize=5.8)
        axis.set_title(family, loc="left", fontsize=9.5, weight="bold", pad=22)
        for boundary in range(len(horizons), len(columns), len(horizons)):
            axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
        for row_index, model in enumerate(models):
            for column_index, rank in enumerate(matrix[row_index]):
                text_color = "white" if rank >= max_rank * 0.65 else "black"
                axis.text(column_index, row_index, f"{int(rank)}", ha="center", va="center", fontsize=3.4, color=text_color)
            if model == "RegimeGatedMamba":
                axis.add_patch(plt.Rectangle((-0.5, row_index - 0.5), len(columns), 1, fill=False, edgecolor="#000000", linewidth=1.1))
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, pad=0.006, fraction=0.018)
    colorbar.set_label("Path-RMSSE rank (1 = best)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    main.suptitle("S-only v2 path-RMSSE rank by target and horizon", x=0.01, ha="left", fontsize=13, weight="bold")
    note_axis = note_band.subplots()
    note_axis.axis("off")
    note_axis.text(
        0,
        0.55,
        "Every cell ranks all 14 models within one target-horizon pair. The outlined row is Regime-Gated Mamba; learned rows are five-seed means.",
        fontsize=7.1,
        va="center",
    )
    return save_figure(figure, output_dir, f"{release}_path_rmsse_rank_by_family")


def plot_v2_regime_gated_position(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Provide a readable target-by-horizon position map for the proposed model."""
    horizons = sorted(table["horizon"].unique())
    tasks = [task for _, family_tasks in v2_family_tasks(table) for task in family_tasks]
    subset = table[table["model_id"] == "RegimeGatedMamba"]
    matrix = np.array(
        [
            [
                subset.loc[(subset["dataset"] == task) & (subset["horizon"] == horizon), "path_rmsse_rank"].iloc[0]
                for horizon in horizons
            ]
            for task in tasks
        ],
        dtype=float,
    )
    max_rank = int(table["path_rmsse_rank"].max())
    cmap = plt.colormaps["viridis_r"].resampled(max_rank)
    figure, axis = plt.subplots(figsize=(8.5, 9.3), layout="constrained")
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        norm=BoundaryNorm(np.arange(0.5, max_rank + 1.5), cmap.N),
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(horizons)), [str(horizon) for horizon in horizons])
    axis.set_yticks(np.arange(len(tasks)), [TASK_ABBREVIATIONS.get(task, task) for task in tasks], fontsize=8)
    axis.set_xlabel("Forecast horizon (retained observations)")
    axis.set_ylabel("Independent target-only series")
    axis.set_title("Regime-Gated Mamba position across the v2 target roster", loc="left", fontsize=12, weight="bold", pad=28)
    axis.text(
        0,
        1.01,
        "Rank 1 is the lowest path RMSSE among 14 models within a target-horizon cell.",
        transform=axis.transAxes,
        fontsize=7.6,
        va="bottom",
    )
    task_offset = 0
    for family, family_tasks in v2_family_tasks(table):
        task_offset += len(family_tasks)
        if task_offset < len(tasks):
            axis.axhline(task_offset - 0.5, color="white", linewidth=2.0)
    for row_index, values in enumerate(matrix):
        for column_index, rank in enumerate(values):
            text_color = "white" if rank >= max_rank * 0.65 else "black"
            axis.text(column_index, row_index, f"{int(rank)}", ha="center", va="center", fontsize=7.2, color=text_color)
    colorbar = figure.colorbar(image, ax=axis, pad=0.01, fraction=0.04)
    colorbar.set_label("Path-RMSSE rank (1 = best)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    return save_figure(figure, output_dir, f"{release}_regime_gated_rank_by_target")


def _v2_task_order(table: pd.DataFrame) -> list[str]:
    return [task for _, family_tasks in v2_family_tasks(table) for task in family_tasks]


def plot_v2_rank_by_horizon(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Show every model's equal-cell rank trajectory for each target family."""
    horizons = sorted(table["horizon"].unique())
    models = model_order(table)
    figure = plt.figure(figsize=(13.2, 8.9), layout="constrained")
    main, legend_band = figure.subfigures(2, 1, height_ratios=[12, 1.7])
    axes = list(main.subplots(2, 2, sharex=True, sharey=True).flat)
    handles: dict[str, plt.Line2D] = {}
    for axis, (family, tasks) in zip(axes, v2_family_tasks(table)):
        subset = table[table["dataset"].isin(tasks)]
        for model in models:
            values = subset[subset["model_id"] == model].groupby("horizon", as_index=False)["path_rmsse_rank"].mean().sort_values("horizon")
            style = dict(MODEL_STYLES[model])
            line = axis.plot(values["horizon"], values["path_rmsse_rank"], markersize=3.8, label=MODEL_LABELS[model], **style)[0]
            handles.setdefault(model, line)
        axis.set_xscale("log")
        axis.set_xticks(horizons)
        axis.set_xticklabels([str(value) for value in horizons])
        axis.invert_yaxis()
        axis.set_ylim(len(models) + 0.45, 0.55)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_title(family, loc="left", fontsize=10, weight="bold")
        axis.set_ylabel("Mean rank (1 = best)")
    for axis in axes[-2:]:
        axis.set_xlabel("Forecast horizon (retained observations; log scale)")
    main.suptitle("V2 path-RMSSE rank trajectories by target family", x=0.01, ha="left", fontsize=13, weight="bold")
    legend_axis = legend_band.subplots()
    legend_axis.axis("off")
    legend_axis.legend(list(handles.values()), [MODEL_LABELS[model] for model in handles], loc="center", ncol=4, fontsize=7, frameon=False)
    return save_figure(figure, output_dir, f"{release}_path_rmsse_rank_by_horizon")


def _v2_metric_matrix(table: pd.DataFrame, metric: str) -> tuple[list[str], list[int], np.ndarray]:
    tasks = _v2_task_order(table)
    horizons = sorted(table["horizon"].unique())
    proposed = table[table["model_id"] == "RegimeGatedMamba"].set_index(["dataset", "horizon"])
    reference = table[table["model_id"] == "LastValue"].set_index(["dataset", "horizon"])
    matrix = np.array(
        [
            [
                float(proposed.loc[(task, horizon), f"path_{metric}_mean"])
                / float(reference.loc[(task, horizon), f"path_{metric}_mean"])
                for horizon in horizons
            ]
            for task in tasks
        ],
        dtype=float,
    )
    return tasks, horizons, matrix


def plot_v2_metric_ratios(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Compare target-level MAE, RMSE, and RMSSE ratios to LastValue."""
    metrics = (("mae", "Path MAE"), ("rmse", "Path RMSE"), ("rmsse", "Path RMSSE"))
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 8.8), sharey=True, layout="constrained")
    image = None
    for axis, (metric, label) in zip(axes, metrics):
        tasks, horizons, ratios = _v2_metric_matrix(table, metric)
        transformed = np.log2(ratios)
        limit = max(0.05, float(np.nanmax(np.abs(transformed))))
        image = axis.imshow(transformed, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest")
        axis.set_xticks(np.arange(len(horizons)), [str(value) for value in horizons], fontsize=7)
        axis.set_yticks(np.arange(len(tasks)), [TASK_ABBREVIATIONS.get(task, task) for task in tasks], fontsize=7)
        axis.set_xlabel("Horizon")
        axis.set_title(label, loc="left", fontsize=10, weight="bold")
        for boundary in np.cumsum([len(tasks) for _, tasks in v2_family_tasks(table)])[:-1]:
            axis.axhline(boundary - 0.5, color="white", linewidth=1.4)
        for row_index, values in enumerate(ratios):
            for column_index, value in enumerate(values):
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=5.2, color="black" if abs(np.log2(value)) < limit * 0.62 else "white")
    assert image is not None
    axes[0].set_ylabel("Independent target-only series")
    colorbar = figure.colorbar(image, ax=axes, pad=0.01, fraction=0.025)
    colorbar.set_label("log2(Regime-Gated Mamba / LastValue); 0 = equal", fontsize=8)
    figure.suptitle("V2 target-level metric ratios relative to LastValue", x=0.01, ha="left", fontsize=13, weight="bold")
    return save_figure(figure, output_dir, f"{release}_regime_gated_metric_ratios")


def plot_v2_baseline_deltas(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Show signed Regime-Gated Mamba path-RMSSE deltas for named references."""
    references = (("LastValue", "LastValue"), ("PatchTST", "PatchTST"), ("TimesNet", "TimesNet"))
    tasks = _v2_task_order(table)
    horizons = sorted(table["horizon"].unique())
    lookup = table.set_index(["dataset", "horizon", "model_id"])
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 8.8), sharey=True, layout="constrained")
    image = None
    for axis, (reference, label) in zip(axes, references):
        matrix = np.array(
            [
                [
                    float(lookup.loc[(task, horizon, "RegimeGatedMamba"), "path_rmsse_mean"])
                    - float(lookup.loc[(task, horizon, reference), "path_rmsse_mean"])
                    for horizon in horizons
                ]
                for task in tasks
            ],
            dtype=float,
        )
        limit = max(0.05, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest")
        axis.set_xticks(np.arange(len(horizons)), [str(value) for value in horizons], fontsize=7)
        axis.set_yticks(np.arange(len(tasks)), [TASK_ABBREVIATIONS.get(task, task) for task in tasks], fontsize=7)
        axis.set_xlabel("Horizon")
        axis.set_title(f"vs {label}", loc="left", fontsize=10, weight="bold")
        for boundary in np.cumsum([len(family_tasks) for _, family_tasks in v2_family_tasks(table)])[:-1]:
            axis.axhline(boundary - 0.5, color="white", linewidth=1.4)
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                axis.text(column_index, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=4.8, color="black" if abs(value) < limit * 0.62 else "white")
    assert image is not None
    axes[0].set_ylabel("Independent target-only series")
    colorbar = figure.colorbar(image, ax=axes, pad=0.01, fraction=0.025)
    colorbar.set_label("RGM minus reference path RMSSE; negative favors RGM", fontsize=8)
    figure.suptitle("V2 Regime-Gated Mamba path-RMSSE deltas by target and horizon", x=0.01, ha="left", fontsize=13, weight="bold")
    return save_figure(figure, output_dir, f"{release}_regime_gated_delta_vs_baselines")


def plot_v2_seed_stability(table: pd.DataFrame, seed_table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Show five-seed normalized RMSSE distributions by family and horizon."""
    last = table[table["model_id"] == "LastValue"][["dataset", "horizon", "path_rmsse_mean"]].rename(columns={"path_rmsse_mean": "last_rmsse"})
    proposed = seed_table[seed_table["model_id"] == "RegimeGatedMamba"][["dataset", "horizon", "path_rmsse"]].merge(last, on=["dataset", "horizon"], how="inner")
    proposed["ratio"] = proposed["path_rmsse"] / proposed["last_rmsse"]
    proposed["family"] = proposed["dataset"].map(lambda task: next(label for label, prefix in V2_FAMILY_SPECS if task.startswith(prefix)))
    horizons = sorted(proposed["horizon"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), sharex=True, sharey=True, layout="constrained")
    for axis, (family, _) in zip(axes.flat, V2_FAMILY_SPECS):
        values = [proposed[(proposed["family"] == family) & (proposed["horizon"] == horizon)]["ratio"].to_numpy() for horizon in horizons]
        box = axis.boxplot(values, positions=np.arange(len(horizons)), widths=0.58, patch_artist=True, showfliers=True, medianprops={"color": "black", "linewidth": 1.3}, whiskerprops={"color": "#4D4D4D"}, capprops={"color": "#4D4D4D"})
        for patch in box["boxes"]:
            patch.set_facecolor("#D9EAF7")
            patch.set_edgecolor("#0072B2")
        axis.axhline(1.0, color="#4D4D4D", linewidth=0.9, linestyle="--")
        axis.set_title(family, loc="left", fontsize=10, weight="bold")
        axis.set_xticks(np.arange(len(horizons)), [str(value) for value in horizons])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_xlabel("Horizon")
        axis.set_ylabel("RGM / LastValue path RMSSE")
    figure.suptitle("V2 five-seed stability relative to LastValue", x=0.01, ha="left", fontsize=13, weight="bold")
    return save_figure(figure, output_dir, f"{release}_regime_gated_seed_stability")


def plot_all_models_by_horizon(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    tasks = sorted(table["dataset"].unique())
    horizons = sorted(table["horizon"].unique())
    models = model_order(table)
    figure = plt.figure(figsize=(12.4, 9.0), layout="constrained")
    main, legend_band = figure.subfigures(2, 1, height_ratios=[12, 1.6])
    axes = main.subplots(2, 2, sharex=True)
    legend_handles = []
    for axis, task in zip(axes.flat, tasks):
        subset = table[table["dataset"] == task]
        for model in models:
            values = subset[subset["model_id"] == model].sort_values("horizon")
            style = dict(MODEL_STYLES[model])
            line = axis.plot(values["horizon"], values["path_rmsse_mean"], label=MODEL_LABELS[model], markersize=4.4, **style)[0]
            if len(legend_handles) < len(models):
                legend_handles.append(line)
        axis.set_xscale("log")
        axis.set_xticks(horizons)
        axis.set_xticklabels([str(value) for value in horizons])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_title(TASK_LABELS.get(task, task), loc="left", fontsize=10, weight="bold")
        axis.set_ylabel("Path RMSSE")
    for axis in axes[-1, :]:
        axis.set_xlabel("Forecast horizon (retained observations; log scale)")
    main.suptitle("S-only v1 path-RMSSE trajectories for every model", x=0.01, ha="left", fontsize=13, weight="bold")
    legend_axis = legend_band.subplots()
    legend_axis.axis("off")
    legend_axis.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="center",
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    return save_figure(figure, output_dir, f"{release}_path_rmsse_by_task_horizon")


def plot_accuracy_efficiency(table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    learned = table[table["seed_count"] > 0]
    summary = (
        learned.groupby("model_id", as_index=False)
        .agg(mean_path_rmsse_rank=("path_rmsse_rank", "mean"), runtime_seconds=("runtime_seconds_mean", "mean"))
        .sort_values("runtime_seconds")
    )
    figure = plt.figure(figsize=(7.6, 5.65), layout="constrained")
    main, note_band = figure.subfigures(2, 1, height_ratios=[10, 1.6])
    axis = main.subplots()
    for row in summary.itertuples(index=False):
        style = MODEL_STYLES[row.model_id]
        axis.scatter(
            row.runtime_seconds,
            row.mean_path_rmsse_rank,
            s=58 if row.model_id == "RegimeGatedMamba" else 42,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.6,
            zorder=4 if row.model_id == "RegimeGatedMamba" else 3,
        )
        offset, alignment = (5, 4), "left"
        if row.model_id == "RegimeGatedMamba":
            offset = (9, -16)
        elif row.model_id == "UngatedRegimeLinearMamba":
            offset = (7, 12)
        elif row.model_id == "MultiScaleDecompositionMamba":
            offset, alignment = (-8, 10), "right"
        elif row.model_id == "MultiScalePatchMamba":
            offset = (8, -14)
        elif row.model_id == "TimesNet":
            offset, alignment = (-8, 7), "right"
        label = "Ungated RLM" if row.model_id == "UngatedRegimeLinearMamba" else MODEL_LABELS[row.model_id]
        axis.annotate(
            label,
            (row.runtime_seconds, row.mean_path_rmsse_rank),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            ha=alignment,
        )
    axis.set_xscale("log")
    axis.invert_yaxis()
    axis.grid(color="#D9D9D9", linewidth=0.6)
    axis.set_xlabel("Mean end-to-end runtime per task-horizon-seed bundle (s; log scale)")
    cell_count = table[["dataset", "horizon"]].drop_duplicates().shape[0]
    axis.set_ylabel(f"Mean path-RMSSE rank across {cell_count} cells\n(lower rank is better)")
    axis.set_title(f"{release_display_name(release)} accuracy position and campaign runtime", loc="left", fontsize=12, weight="bold")
    note_axis = note_band.subplots()
    note_axis.axis("off")
    note_axis.text(
        0,
        0.55,
        "Runtime includes each horizon's training and evaluation run; deterministic references are excluded because they have no seed-level training time.",
        fontsize=7,
        va="center",
    )
    return save_figure(figure, output_dir, f"{release}_accuracy_efficiency")


def plot_v2_gate_ablation(bootstrap_table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    """Plot the prespecified, release-wide gate-ablation contrast with its intervals."""
    subset = bootstrap_table.loc[
        (bootstrap_table["model_id"] == "RegimeGatedMamba")
        & (bootstrap_table["reference_model_id"] == "UngatedRegimeLinearMamba")
    ].sort_values("horizon")
    if subset.empty:
        raise RuntimeError("V2 release is missing the Regime-Gated Mamba ablation contrast")
    values = subset["mean_path_rmsse_difference"].to_numpy(dtype=float)
    lower = subset["bootstrap_ci_low"].to_numpy(dtype=float)
    upper = subset["bootstrap_ci_high"].to_numpy(dtype=float)
    significant = subset["p_value_holm"].to_numpy(dtype=float) < 0.05
    horizons = subset["horizon"].to_numpy(dtype=int)
    figure = plt.figure(figsize=(7.8, 5.5), layout="constrained")
    main, note_band = figure.subfigures(2, 1, height_ratios=[10, 1.8])
    axis = main.subplots()
    axis.axhline(0, color="#4D4D4D", linewidth=0.9, zorder=1)
    axis.errorbar(
        horizons,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        color="#000000",
        marker="D",
        markersize=5,
        linewidth=1.35,
        capsize=3,
        label="All horizons",
        zorder=3,
    )
    axis.scatter(
        horizons[significant],
        values[significant],
        s=39,
        marker="D",
        color="#000000",
        edgecolor="white",
        linewidth=0.65,
        label="Holm-adjusted p < 0.05",
        zorder=4,
    )
    axis.set_xscale("log")
    axis.set_xticks(horizons)
    axis.set_xticklabels([str(value) for value in horizons])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_xlabel("Forecast horizon (retained observations; log scale)")
    axis.set_ylabel("Regime-Gated Mamba minus ungated\nmean path RMSSE (negative favors gate)")
    axis.set_title("V2 gate-ablation contrast across 18 target clusters", loc="left", fontsize=12, weight="bold")
    axis.legend(loc="best", frameon=False, fontsize=7.5)
    note_axis = note_band.subplots()
    note_axis.axis("off")
    note_axis.text(
        0,
        0.55,
        "Points are observed paired differences; bars are 95% nested task/seed/moving-origin block-bootstrap intervals (10,000 replicates).",
        fontsize=7.1,
        va="center",
    )
    return save_figure(figure, output_dir, f"{release}_gate_ablation_bootstrap")


def learned_forecast(root: Path, seed_table: pd.DataFrame, task: str, model_id: str, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = seed_table.loc[
        (seed_table["dataset"] == task)
        & (seed_table["model_id"] == model_id)
        & (seed_table["seed"].astype(str) == "2023")
        & (seed_table["horizon"] == horizon)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing seed-2023 forecast for {task}/{model_id}/h{horizon}")
    run_path = REPO_ROOT / "results" / "financial_s_only_retrospective_v1" / "runs" / row.iloc[0]["setting"]
    origins = np.load(run_path / "origin_dates.npy", allow_pickle=False).astype(str)
    dates = np.load(run_path / "forecast_dates.npy", allow_pickle=False).astype(str)
    prediction = np.load(run_path / "pred.npy", allow_pickle=False)
    truth = np.load(run_path / "true.npy", allow_pickle=False)
    return origins, dates, prediction, truth


def arima_forecast(root: Path, task: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    metrics_path = root / "metrics" / "deterministic_metrics.csv"
    table = pd.read_csv(metrics_path)
    row = table.loc[(table["dataset"] == task) & (table["model_id"] == "ARIMA") & (table["horizon"] == horizon)]
    if len(row) != 1:
        raise RuntimeError(f"Missing ARIMA forecast for {task}/h{horizon}")
    bundle = REPO_ROOT / row.iloc[0]["bundle"]
    payload = np.load(bundle, allow_pickle=False)
    return payload["origin_dates"].astype(str), payload["prediction"]


def plot_forecast_examples(root: Path, seed_table: pd.DataFrame, output_dir: Path, release: str) -> list[Path]:
    if release != "financial_s_only_v1":
        return []
    manifest = json.loads((root / "training_datasets.json").read_text(encoding="utf-8"))
    tasks = sorted(manifest)
    horizon = 252
    figure = plt.figure(figsize=(12.4, 8.55), layout="constrained")
    main, legend_band = figure.subfigures(2, 1, height_ratios=[12, 1.6])
    axes = main.subplots(2, 2)
    legend_handles: dict[str, plt.Line2D] = {}
    for axis, task in zip(axes.flat, tasks):
        origins, forecast_dates, _, truth = learned_forecast(root, seed_table, task, "RegimeGatedMamba", horizon)
        final_origin = origins[-1]
        entry = manifest[task]
        frame = pd.read_csv(REPO_ROOT / entry["root_path"] / entry["data_path"], usecols=["date", entry["target"]])
        dates = pd.to_datetime(frame["date"], errors="raise")
        origin_index = int(np.flatnonzero(dates.astype(str).to_numpy() == final_origin)[0])
        history_start = max(0, origin_index - 96)
        history_line = axis.plot(dates.iloc[history_start:origin_index + 1], frame[entry["target"]].iloc[history_start:origin_index + 1], color="#4D4D4D", linewidth=1.6, label="Observed history")[0]
        truth_dates = pd.to_datetime(forecast_dates[-1])
        truth_line = axis.plot(truth_dates, truth[-1, :, 0], color="#000000", linewidth=1.8, linestyle="--", label="Observed future")[0]
        legend_handles.setdefault("Observed history", history_line)
        legend_handles.setdefault("Observed future", truth_line)
        for model_id in FORECAST_MODEL_IDS:
            model_origins, model_dates, prediction, _ = learned_forecast(root, seed_table, task, model_id, horizon)
            index = int(np.flatnonzero(model_origins == final_origin)[0])
            line = axis.plot(pd.to_datetime(model_dates[index]), prediction[index, :, 0], color=FORECAST_COLORS[model_id], linewidth=1.3, label=MODEL_LABELS[model_id])[0]
            legend_handles.setdefault(MODEL_LABELS[model_id], line)
        arima_origins, arima_prediction = arima_forecast(root, task, horizon)
        arima_index = int(np.flatnonzero(arima_origins == final_origin)[0])
        arima_line = axis.plot(truth_dates, arima_prediction[arima_index], color=FORECAST_COLORS["ARIMA"], linewidth=1.2, label="ARIMA")[0]
        legend_handles.setdefault("ARIMA", arima_line)
        axis.axvline(pd.Timestamp(str(final_origin)), color="#7F7F7F", linewidth=0.8)
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_title(TASK_LABELS.get(task, task), loc="left", fontsize=10, weight="bold")
        axis.set_ylabel(entry["target"] + " level")
    for axis in axes[-1, :]:
        axis.set_xlabel("Date")
    main.suptitle("Illustrative latest-origin 252-observation forecasts", x=0.01, ha="left", fontsize=13, weight="bold")
    legend_axis = legend_band.subplots()
    legend_axis.axis("off")
    legend_axis.legend(list(legend_handles.values()), list(legend_handles), loc="center", ncol=3, fontsize=8, frameon=False)
    return save_figure(figure, output_dir, f"{release}_illustrative_forecasts_h252")


def render(release: str, output_dir: Path) -> dict:
    root, aggregate_table, seed_table, status = load_release(release)
    bootstrap_path = root / "metrics" / "paired_block_bootstrap.csv"
    bootstrap_table = pd.read_csv(bootstrap_path) if bootstrap_path.is_file() else pd.DataFrame()
    outputs = []
    if release == "financial_s_only_v1":
        outputs.extend(plot_rank_heatmap(aggregate_table, output_dir, release))
        outputs.extend(plot_all_models_by_horizon(aggregate_table, output_dir, release))
        outputs.extend(plot_accuracy_efficiency(aggregate_table, output_dir, release))
        outputs.extend(plot_forecast_examples(root, seed_table, output_dir, release))
        figure_descriptions = {
            f"{release}_path_rmsse_rank_heatmap": {
                "description": "All model path-RMSSE ranks for every v1 task-horizon cell.",
                "alt_text": "A heatmap of ranks one through fourteen for every model across four task groups and six horizons.",
            },
            f"{release}_path_rmsse_by_task_horizon": {
                "description": "All model path-RMSSE trajectories within each v1 target-only task.",
                "alt_text": "Four line charts, one per S-only task, with path RMSSE across six forecast horizons for all models.",
            },
            f"{release}_accuracy_efficiency": {
                "description": "Learned-model mean rank versus campaign runtime.",
                "alt_text": "A log-runtime scatterplot of learned model accuracy rank, with lower rank better.",
            },
            f"{release}_illustrative_forecasts_h252": {
                "description": "Latest-origin h252 trajectory examples, seed 2023 for learned models.",
                "alt_text": "Four time-series panels showing observed history and future alongside selected model forecasts at the latest h252 origin.",
            },
        }
    else:
        outputs.extend(plot_v2_rank_heatmap(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_regime_gated_position(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_rank_by_horizon(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_metric_ratios(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_baseline_deltas(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_seed_stability(aggregate_table, seed_table, output_dir, release))
        outputs.extend(plot_accuracy_efficiency(aggregate_table, output_dir, release))
        outputs.extend(plot_v2_gate_ablation(bootstrap_table, output_dir, release))
        figure_descriptions = {
            f"{release}_path_rmsse_rank_by_family": {
                "description": "All model path-RMSSE ranks across every v2 target and horizon, separated by data family.",
                "alt_text": "Four heatmaps showing fourteen model ranks for each independent financial target and horizon in the v2 release.",
            },
            f"{release}_regime_gated_rank_by_target": {
                "description": "Regime-Gated Mamba path-RMSSE rank across all 18 v2 targets and six horizons.",
                "alt_text": "An eighteen-by-six heatmap showing the proposed model's rank from one through fourteen for each target and horizon.",
            },
            f"{release}_path_rmsse_rank_by_horizon": {
                "description": "Equal-cell model rank trajectories by horizon and target family.",
                "alt_text": "Four line-chart panels showing every model's mean path-RMSSE rank across six horizons for each target family.",
            },
            f"{release}_regime_gated_metric_ratios": {
                "description": "Target-level path MAE, RMSE, and RMSSE ratios relative to LastValue.",
                "alt_text": "Three target-by-horizon heatmaps of log two ratios for path MAE, RMSE, and RMSSE, where zero means equal to LastValue.",
            },
            f"{release}_regime_gated_delta_vs_baselines": {
                "description": "Signed Regime-Gated Mamba path-RMSSE deltas versus LastValue, PatchTST, and TimesNet.",
                "alt_text": "Three target-by-horizon heatmaps of proposed-model minus reference path RMSSE, with negative values favoring the proposed model.",
            },
            f"{release}_regime_gated_seed_stability": {
                "description": "Five-seed normalized path-RMSSE distributions relative to LastValue by family and horizon.",
                "alt_text": "Four boxplot panels showing five-seed Regime-Gated Mamba to LastValue path-RMSSE ratios across target families and horizons.",
            },
            f"{release}_accuracy_efficiency": {
                "description": "Learned-model mean rank across 108 v2 cells versus campaign runtime.",
                "alt_text": "A log-runtime scatterplot of learned model accuracy rank across one hundred and eight cells, with lower rank better.",
            },
            f"{release}_gate_ablation_bootstrap": {
                "description": "The prespecified Regime-Gated Mamba versus ungated ablation contrast by horizon.",
                "alt_text": "A point-and-interval plot of the gated-minus-ungated path RMSSE difference, with negative values favoring the gate.",
            },
        }
    source_paths = [
        root / "metrics" / "metrics_aggregate.csv",
        root / "metrics" / "metrics_by_seed.csv",
        root / "metrics" / "deterministic_metrics.csv",
        root / "release_status.json",
    ]
    if bootstrap_path.is_file():
        source_paths.append(bootstrap_path)
    manifest = {
        "release": release,
        "release_status": status["status"],
        "source_files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)} for path in source_paths],
        "output_files": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)} for path in outputs],
        "figures": figure_descriptions,
        "notes": [
            "All rank and metric figures use the checksum-covered release tables.",
            "The forecast trajectory display is illustrative and is not used for aggregate performance claims.",
            "Raw MAE and RMSE remain task-specific; normalized ratio figures use an explicit named reference.",
        ],
    }
    manifest_path = output_dir / f"{release}_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"release": release, "figure_count": len(outputs), "outputs": [str(path.relative_to(REPO_ROOT)) for path in outputs], "manifest": str(manifest_path.relative_to(REPO_ROOT))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="financial_s_only_v1", choices=("financial_s_only_v1", "financial_s_only_v2"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args.release, args.output_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
