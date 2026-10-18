#!/usr/bin/env python
"""Create provenance-backed rolling forecast-versus-observed financial figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.registry import public_name


CONFIG_PATH = REPO_ROOT / "configs" / "financial_mamba_rerun_v1.json"
OUTPUT_ROOT = REPO_ROOT / "results" / "financial_mamba_rerun_v1"
SUMMARY_NAME = "campaign_summary.json"
MODEL_COLORS = {
    "RegimeGatedMamba": "#0072B2",
    "MultiScalePatchMamba": "#E69F00",
    "InvertedMamba": "#009E73",
    "DecompositionMamba": "#CC79A7",
    "MultiScaleDecompositionMamba": "#D55E00",
}
MODEL_STYLES = {
    "RegimeGatedMamba": "-",
    "MultiScalePatchMamba": "--",
    "InvertedMamba": ":",
    "DecompositionMamba": "-.",
    "MultiScaleDecompositionMamba": (0, (3, 1, 1, 1)),
}
TASK_LABELS = {
    "financial_cross_asset": "Cross-asset (GLD)",
    "financial_crypto": "Cryptoasset (BTCUSDT)",
    "financial_fx": "Foreign exchange (GBPUSD)",
    "financial_yields": "Treasury yields (DGS10)",
}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected(values: list[str], available: list[str], label: str) -> list[str]:
    if values == ["ALL"]:
        return list(available)
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; available={available}")
    return list(values)


def scope(config: dict, args: argparse.Namespace) -> tuple[list[str], list[str], list[int], list[int]]:
    manifest = json.loads((REPO_ROOT / config["base_protocol"]["training_manifest"]).read_text(encoding="utf-8"))
    tasks = selected(args.tasks, sorted(manifest), "task IDs")
    models = selected(args.models, list(config["models"]), "model IDs")
    available_horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    horizons = [int(value) for value in selected([str(value) for value in (args.horizons or available_horizons)], [str(value) for value in available_horizons], "horizons")]
    available_seeds = [int(value) for value in config["base_protocol"]["seeds"]]
    seeds = [int(value) for value in selected([str(value) for value in args.seeds], [str(value) for value in available_seeds], "seeds")]
    return tasks, models, horizons, seeds


def load_terminal_series(summary: dict, output_root: Path, task: str, model_id: str, horizon: int, seeds: list[int]) -> dict:
    predictions, truth, dates = [], None, None
    for seed in seeds:
        record = summary.get("runs", {}).get(f"{task}|{model_id}|{seed}", {})
        setting = record.get("settings", {}).get(str(horizon))
        directory = output_root / "runs" / str(setting)
        if record.get("status") != "complete" or not setting:
            raise RuntimeError(f"Missing completed run for {task}/{model_id}/seed={seed}/h={horizon}")
        prediction = np.load(directory / "pred.npy", mmap_mode="r")[:, -1, -1].astype(float)
        observed = np.load(directory / "true.npy", mmap_mode="r")[:, -1, -1].astype(float)
        forecast_dates = np.load(directory / "forecast_dates.npy", allow_pickle=False)[:, -1].astype(str)
        if truth is None:
            truth, dates = observed, forecast_dates
        elif not np.array_equal(truth, observed) or not np.array_equal(dates, forecast_dates):
            raise ValueError(f"Seed artifacts disagree for {task}/{model_id}/h={horizon}")
        predictions.append(prediction)
    stacked = np.stack(predictions, axis=0)
    return {
        "dates": pd.to_datetime(dates),
        "observed": truth,
        "seed_predictions": stacked,
        "mean": np.mean(stacked, axis=0),
        "q05": np.quantile(stacked, 0.05, axis=0),
        "q95": np.quantile(stacked, 0.95, axis=0),
    }


def write_trace(path: Path, series: dict, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["target_date", "observed", *[f"seed_{seed}" for seed in seeds], "mean_prediction", "q05_prediction", "q95_prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, date in enumerate(series["dates"]):
            row = {
                "target_date": date.strftime("%Y-%m-%d"),
                "observed": series["observed"][index],
                "mean_prediction": series["mean"][index],
                "q05_prediction": series["q05"][index],
                "q95_prediction": series["q95"][index],
            }
            row.update({f"seed_{seed}": series["seed_predictions"][position, index] for position, seed in enumerate(seeds)})
            writer.writerow(row)


def value_limits(series_items: list[dict]) -> tuple[float, float]:
    values = np.concatenate([np.concatenate((item["observed"], item["q05"], item["q95"])) for item in series_items])
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    margin = max((upper - lower) * 0.03, abs(upper) * 0.01, 1e-9)
    return lower - margin, upper + margin


def style_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def render_individual(path: Path, task: str, model_id: str, public_name: str, horizon: int, series: dict, limits: tuple[float, float], seeds: list[int]) -> None:
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10}):
        figure, axis = plt.subplots(figsize=(7.0, 3.5), layout="constrained", facecolor="white")
        axis.plot(series["dates"], series["observed"], color="#1A1A1A", linewidth=1.2, label="Observed target")
        axis.fill_between(series["dates"], series["q05"], series["q95"], color=MODEL_COLORS[model_id], alpha=0.20, linewidth=0, label="Seed 5th-95th percentile")
        axis.plot(series["dates"], series["mean"], color=MODEL_COLORS[model_id], linestyle=MODEL_STYLES[model_id], linewidth=1.2, label=f"{public_name} mean")
        axis.set(
            title=f"{TASK_LABELS[task.rsplit('_', 1)[0]]}, {task.rsplit('_', 1)[1].upper()} input, horizon {horizon}",
            xlabel="Forecast target date",
            ylabel="Target level",
            ylim=limits,
        )
        style_axis(axis)
        axis.legend(frameon=False, loc="best")
        figure.text(0.01, 0.01, f"Five independent training seeds ({', '.join(map(str, seeds))}); no smoothing.", fontsize=7, color="#404040")
        figure.savefig(path, dpi=300, facecolor="white")
        plt.close(figure)


def render_summary(path_base: Path, horizon: int, tasks: list[str], models: list[str], public_names: dict[str, str], series: dict[tuple[str, str, int], dict]) -> list[Path]:
    families = ["financial_cross_asset", "financial_crypto", "financial_fx", "financial_yields"]
    active_families = [family for family in families if f"{family}_s" in tasks and f"{family}_ms" in tasks]
    if not active_families:
        return []
    path_base.parent.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8}):
        figure, axes = plt.subplots(len(active_families), 2, figsize=(7.2, 2.0 * len(active_families)), layout="constrained", sharex=False, facecolor="white")
        axes = np.atleast_2d(axes)
        for row, family in enumerate(active_families):
            family_series = [series[(f"{family}_{view}", model, horizon)] for view in ("s", "ms") for model in models]
            limits = value_limits(family_series)
            for column, view in enumerate(("s", "ms")):
                task = f"{family}_{view}"
                axis = axes[row, column]
                reference = series[(task, models[0], horizon)]
                axis.plot(reference["dates"], reference["observed"], color="#1A1A1A", linewidth=1.0, label="Observed" if row == 0 and column == 0 else None)
                for model in models:
                    current = series[(task, model, horizon)]
                    axis.plot(current["dates"], current["mean"], color=MODEL_COLORS[model], linestyle=MODEL_STYLES[model], linewidth=0.9, label=public_names[model] if row == 0 and column == 0 else None)
                axis.set(title=f"{TASK_LABELS[family]} ({view.upper()})", ylabel="Target level", ylim=limits)
                style_axis(axis)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))
        figure.suptitle(f"Rolling terminal forecasts at horizon {horizon}", y=1.10, fontsize=10)
        outputs = []
        for suffix in ("png", "pdf", "svg"):
            output = path_base.with_suffix(f".{suffix}")
            figure.savefig(output, dpi=300 if suffix == "png" else None, facecolor="white")
            outputs.append(output)
        plt.close(figure)
        return outputs


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    tasks, models, horizons, seeds = scope(config, args)
    summary_path = args.output_root / SUMMARY_NAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing campaign summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") != config["protocol_version"]:
        raise ValueError("Campaign summary protocol does not match figure configuration")
    figure_root = args.figure_root
    if figure_root.exists() and any(figure_root.iterdir()) and not args.force:
        raise RuntimeError(f"Figure root already contains files: {figure_root}; pass --force to replace plots")
    figure_root.mkdir(parents=True, exist_ok=True)
    public_names = {model: public_name(model) for model in models}
    series = {}
    outputs = []
    for task in tasks:
        for horizon in horizons:
            task_series = []
            for model in models:
                current = load_terminal_series(summary, args.output_root, task, model, horizon, seeds)
                series[(task, model, horizon)] = current
                task_series.append(current)
            limits = value_limits(task_series)
            for model in models:
                current = series[(task, model, horizon)]
                trace_path = figure_root / "data" / task / model / f"horizon_{horizon}.csv"
                figure_path = figure_root / "rolling" / task / model / f"horizon_{horizon}.png"
                write_trace(trace_path, current, seeds)
                figure_path.parent.mkdir(parents=True, exist_ok=True)
                render_individual(figure_path, task, model, public_names[model], horizon, current, limits, seeds)
                outputs.append({"figure": str(figure_path.relative_to(args.output_root)), "data": str(trace_path.relative_to(args.output_root))})
    summary_outputs = []
    for horizon in [value for value in (60, 252) if value in horizons]:
        if all((task, model, horizon) in series for task in tasks for model in models):
            summary_outputs.extend(render_summary(figure_root / "summary" / f"mamba_comparison_h{horizon}", horizon, tasks, models, public_names, series))
    manifest = {
        "protocol_version": config["protocol_version"],
        "evidence_status": config["evidence_status"],
        "source_summary": str(summary_path.relative_to(REPO_ROOT)),
        "source_summary_sha256": sha256_path(summary_path),
        "tasks": tasks,
        "models": models,
        "horizons": horizons,
        "seeds": seeds,
        "estimator": "mean terminal prediction across independent training seeds",
        "uncertainty": "empirical 5th-95th percentile across independent training seeds",
        "transformations": ["select terminal step from each saved multi-step forecast", "no smoothing", "common y limits within task and horizon"],
        "individual_figures": outputs,
        "summary_figures": [str(path.relative_to(args.output_root)) for path in summary_outputs],
        "journal_status": "publication-neutral provisional figures; journal-specific formatting not yet selected",
    }
    atomic_write_json(figure_root / "figure_manifest.json", manifest)
    return {"individual_figures": len(outputs), "summary_files": len(summary_outputs), "manifest": str((figure_root / "figure_manifest.json").relative_to(REPO_ROOT))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=["ALL"])
    parser.add_argument("--models", nargs="+", default=["ALL"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025, 2026, 2027])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.figure_root = (args.figure_root or args.output_root / "figures").resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Forecast plotting blocked: {exc}") from None
