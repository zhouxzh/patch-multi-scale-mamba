#!/usr/bin/env python
"""Generate final uncertainty summaries and figures from frozen evaluation metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "metrics"
FIGURES = REPO_ROOT / "docs" / "figures"
SEED = 20260809
REPLICATES = 10_000
from utils.financial.registry import public_name

CANDIDATE = "RegimeGatedMamba"


def nested_task_seed_bootstrap(differences: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample tasks then seeds; intervals are descriptive because task views overlap."""
    task_values = {
        task: group.sort_values("seed").difference.to_numpy(dtype=float)
        for task, group in differences.groupby("dataset", sort=True)
    }
    tasks = list(task_values)
    values = np.empty(REPLICATES, dtype=float)
    for replicate in range(REPLICATES):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled = [rng.choice(task_values[task], size=len(task_values[task]), replace=True) for task in sampled_tasks]
        values[replicate] = float(np.mean(np.concatenate(sampled)))
    return values


def bootstrap_contrasts() -> pd.DataFrame:
    data = pd.read_csv(METRICS / "balanced_path_rmsse_by_task_seed.csv")
    candidate = data[data.model_id.eq(CANDIDATE)][["dataset", "seed", "balanced_path_rmsse"]]
    rows = []
    for model_id in sorted(set(data.model_id) - {CANDIDATE}):
        reference = data[data.model_id.eq(model_id)][["dataset", "seed", "model", "balanced_path_rmsse"]]
        paired = candidate.merge(reference, on=["dataset", "seed"], suffixes=("_candidate", "_reference"))
        if len(paired) != 40:
            raise RuntimeError(f"Expected 40 paired task-seed summaries for {model_id}, found {len(paired)}")
        paired["difference"] = paired.balanced_path_rmsse_candidate - paired.balanced_path_rmsse_reference
        rng = np.random.default_rng(SEED + sum(map(ord, model_id)))
        samples = nested_task_seed_bootstrap(paired, rng)
        reference_mean = float(paired.balanced_path_rmsse_reference.mean())
        difference = float(paired.difference.mean())
        rows.append({
            "candidate_id": CANDIDATE,
            "candidate": public_name(CANDIDATE),
            "reference_id": model_id,
            "reference": paired.model.iloc[0],
            "paired_task_seed_summaries": len(paired),
            "mean_balanced_path_rmsse_difference": difference,
            "relative_improvement_pct": -100.0 * difference / reference_mean,
            "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
            "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed": SEED,
            "replication_unit": "nested task then seed; descriptive because S/MS task views overlap",
        })
    result = pd.DataFrame(rows).sort_values("mean_balanced_path_rmsse_difference")
    result.to_csv(METRICS / "regime_gated_mamba_paired_task_seed_bootstrap.csv", index=False)
    return result


def figures() -> None:
    aggregate = pd.read_csv(METRICS / "all_model_aggregate_metrics.csv")
    table = aggregate.pivot_table(index=["dataset", "horizon"], columns="model_id", values="path_rmsse_mean")
    ranks = table.rank(axis=1, method="min")
    mean_by_horizon = ranks.groupby(level="horizon").mean().T
    model_names = aggregate.drop_duplicates("model_id").set_index("model_id").model.to_dict()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), layout="constrained")
    horizons = mean_by_horizon.columns.to_list()
    horizon_positions = np.arange(len(horizons))
    for model_id, values in mean_by_horizon.sort_values(1).iterrows():
        is_candidate = model_id == CANDIDATE
        ax.plot(horizon_positions, values.reindex(horizons).to_numpy(), marker="o" if is_candidate else None,
                linewidth=2.6 if is_candidate else 1.0, alpha=1.0 if is_candidate else 0.55,
                color="#0072B2" if is_candidate else "#666666",
                label=model_names[model_id] if is_candidate else None)
    ax.invert_yaxis()
    ax.set(xlabel="Forecast horizon", ylabel="Mean path-RMSSE rank (lower is better)")
    ax.set_xticks(horizon_positions, horizons)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIGURES / "regime_gated_mamba_rank_by_horizon.png", dpi=300)
    fig.savefig(FIGURES / "regime_gated_mamba_rank_by_horizon.pdf")
    plt.close(fig)

    candidate_rank = ranks[CANDIDATE].unstack("horizon").sort_index()
    fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
    image = ax.imshow(candidate_rank.to_numpy(), cmap="YlGnBu_r", vmin=1, vmax=len(table.columns), aspect="auto")
    ax.set(xticks=np.arange(len(candidate_rank.columns)), xticklabels=candidate_rank.columns,
           yticks=np.arange(len(candidate_rank.index)), yticklabels=candidate_rank.index,
           xlabel="Forecast horizon", ylabel="Task")
    for row in range(candidate_rank.shape[0]):
        for column in range(candidate_rank.shape[1]):
            value = candidate_rank.iat[row, column]
            ax.text(column, row, f"{value:.0f}", ha="center", va="center",
                    color="white" if value >= 6 else "black")
    fig.colorbar(image, ax=ax, label="Regime-Gated Mamba path-RMSSE rank")
    fig.savefig(FIGURES / "regime_gated_mamba_rank_by_task_horizon.png", dpi=300)
    fig.savefig(FIGURES / "regime_gated_mamba_rank_by_task_horizon.pdf")
    plt.close(fig)

    contrasts = pd.read_csv(METRICS / "regime_gated_mamba_paired_task_seed_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.0), layout="constrained")
    shown = contrasts.sort_values("relative_improvement_pct")
    colors = ["#0072B2" if low <= 0 <= high else "#999999"
              for low, high in zip(shown.bootstrap_ci_low, shown.bootstrap_ci_high)]
    ax.barh(shown.reference, shown.relative_improvement_pct, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Regime-Gated Mamba improvement in balanced path RMSSE (%)")
    fig.savefig(FIGURES / "regime_gated_mamba_bootstrap_contrasts.png", dpi=300)
    fig.savefig(FIGURES / "regime_gated_mamba_bootstrap_contrasts.pdf")
    plt.close(fig)

    profile_path = METRICS / "matched_cuda_microbenchmark.csv"
    if profile_path.is_file():
        profile = pd.read_csv(profile_path)
        average_rank = ranks.mean().rename("mean_rank")
        profile["mean_rank"] = profile.model_id.map(average_rank)
        fig, ax = plt.subplots(figsize=(5.8, 4.0), layout="constrained")
        for row in profile.itertuples():
            ax.scatter(row.median_latency_ms, row.mean_rank, s=55,
                       color="#0072B2" if row.model == public_name(CANDIDATE) else "#666666")
            ax.annotate(row.model, (row.median_latency_ms, row.mean_rank), xytext=(5, 4),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set(xlabel="Matched CUDA median latency (ms, log scale)",
               ylabel="Mean path-RMSSE rank (lower is better)")
        fig.savefig(FIGURES / "regime_gated_mamba_accuracy_efficiency.png", dpi=300)
        fig.savefig(FIGURES / "regime_gated_mamba_accuracy_efficiency.pdf")
        plt.close(fig)


def main() -> None:
    contrasts = bootstrap_contrasts()
    figures()
    manifest = {
        "protocol_version": "financial_reference_evaluation_v1",
        "candidate": CANDIDATE,
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "contrast_count": len(contrasts),
        "source_metrics": [
            "balanced_path_rmsse_by_task_seed.csv",
            "all_model_aggregate_metrics.csv",
        ],
        "figures": [
            "regime_gated_mamba_rank_by_horizon.png",
            "regime_gated_mamba_rank_by_task_horizon.png",
            "regime_gated_mamba_bootstrap_contrasts.png",
            "regime_gated_mamba_accuracy_efficiency.png",
        ],
    }
    (METRICS / "final_evaluation_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
