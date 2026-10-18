#!/usr/bin/env python
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

from utils.financial.common import DEFAULT_PROTOCOL, MANIFEST_DIR, atomic_write_json, load_protocol


TRAINING_MANIFEST = MANIFEST_DIR / "training_datasets.json"
FINAL_SUMMARY = REPO_ROOT / "results" / "financial_confirmatory" / "final_runs.json"
RUN_ROOT = REPO_ROOT / "results" / "financial_confirmatory" / "runs"
STATISTICAL_METRICS = REPO_ROOT / "results" / "financial_confirmatory" / "statistical" / "metrics.csv"
ANALYSIS_ROOT = REPO_ROOT / "results" / "financial_confirmatory" / "analysis"
MAMBA_PROPOSALS = ("DecompositionMamba", "MultiScaleDecompositionMamba", "MultiScalePatchMamba")
NON_MAMBA = ("DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer")
STATISTICAL_COMMON = ("Last-value", "Random-walk-drift", "ARIMA")


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the frozen financial benchmark.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=TRAINING_MANIFEST)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def metrics(truth: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    error = truth - prediction
    mse = float(np.mean(error ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "rmsse": math.sqrt(mse / scale),
    }


def expected_run_keys(protocol: dict, task_ids: list[str]) -> set[str]:
    return {
        f"{task}|{model}|{seed}"
        for task in task_ids
        for model in protocol["level_models"]
        for seed in protocol["final_seeds"]
    }


def target_scales(manifest: dict, protocol: dict) -> dict[str, float]:
    scales = {}
    for task, config in manifest.items():
        frame = pd.read_csv(REPO_ROOT / config["root_path"] / config["data_path"])
        train = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(protocol["splits"]["train_end"])]
        differences = np.diff(train[config["target"]].to_numpy(dtype=float))
        scale = float(np.mean(differences ** 2))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid RMSSE scale for {task}")
        scales[task] = scale
    return scales


def expected_origins(manifest: dict, protocol: dict) -> dict[tuple[str, int], np.ndarray]:
    expected = {}
    start = pd.Timestamp(protocol["splits"]["test_start"])
    end = pd.Timestamp(protocol["splits"]["test_end"])
    for task, config in manifest.items():
        frame = pd.read_csv(REPO_ROOT / config["root_path"] / config["data_path"], usecols=["date"])
        dates = pd.to_datetime(frame["date"], errors="raise")
        dates = dates[(dates >= start) & (dates <= end)].dt.strftime("%Y-%m-%d").to_numpy()
        for horizon in protocol["horizons"]:
            expected[(task, int(horizon))] = dates[: len(dates) - int(horizon) + 1]
    return expected


def require_artifact(path: Path, description: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing {description}: {path}")


def load_neural(protocol: dict, manifest: dict) -> tuple[pd.DataFrame, dict]:
    require_artifact(FINAL_SUMMARY, "final neural summary")
    payload = json.loads(FINAL_SUMMARY.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("Final neural summary uses a different protocol version")
    protocol_hashes = {config["protocol_sha256"] for config in manifest.values()}
    if len(protocol_hashes) != 1 or payload.get("protocol_sha256") != protocol_hashes.pop():
        raise RuntimeError("Final neural summary uses a different protocol hash")
    if payload.get("horizons") != protocol["horizons"]:
        raise RuntimeError("Final neural summary uses a different horizon suite")
    if payload.get("seeds") != protocol["final_seeds"]:
        raise RuntimeError("Final neural summary uses a different seed suite")
    runs = payload.get("runs", {})
    expected = expected_run_keys(protocol, sorted(manifest))
    complete = {key for key, value in runs.items() if value.get("status") == "complete"}
    if complete != expected:
        raise RuntimeError(
            f"Incomplete final neural matrix: complete={len(complete)}, expected={len(expected)}, "
            f"missing={len(expected - complete)}, extra={len(complete - expected)}"
        )
    scales = target_scales(manifest, protocol)
    origins_expected = expected_origins(manifest, protocol)
    rows = []
    losses = {}
    origins = {}
    for run_key in sorted(expected):
        run = runs[run_key]
        task, model, seed_text = run_key.split("|")
        seed = int(seed_text)
        if set(map(int, run.get("settings", {}))) != set(protocol["horizons"]):
            raise ValueError(f"Incomplete horizon settings for {run_key}")
        for horizon in protocol["horizons"]:
            setting = run["settings"][str(horizon)] if str(horizon) in run["settings"] else run["settings"][horizon]
            directory = RUN_ROOT / setting
            prediction = np.load(directory / "pred.npy")
            truth = np.load(directory / "true.npy")
            origin_dates = np.load(directory / "origin_dates.npy", allow_pickle=False).astype(str)
            metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            if prediction.shape != truth.shape or prediction.shape[1] != horizon:
                raise ValueError(f"Prediction shape mismatch for {run_key}/h{horizon}")
            if prediction.shape[0] != len(origin_dates):
                raise ValueError(f"Origin count mismatch for {run_key}/h{horizon}")
            if not np.array_equal(origin_dates, origins_expected[(task, horizon)]):
                raise ValueError(f"Test origins mismatch for {run_key}/h{horizon}")
            expected_metadata = {
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": manifest[task]["protocol_sha256"],
                "dataset_sha256": manifest[task]["data_sha256"],
                "model_label": model,
                "prediction_horizon": horizon,
                "evaluation_split": "test",
                "seed": seed,
                "checkpoint_retained": False,
            }
            for field, expected_value in expected_metadata.items():
                if metadata.get(field) != expected_value:
                    raise ValueError(
                        f"Metadata mismatch for {run_key}/h{horizon}: "
                        f"{field}={metadata.get(field)!r}, expected={expected_value!r}"
                    )
            runtime_seconds = float(metadata.get("runtime_seconds", float("nan")))
            peak_gpu_memory_mb = float(metadata.get("peak_gpu_memory_mb", float("nan")))
            if not np.isfinite(runtime_seconds) or runtime_seconds <= 0:
                raise ValueError(f"Invalid runtime metadata for {run_key}/h{horizon}")
            if not np.isfinite(peak_gpu_memory_mb) or peak_gpu_memory_mb <= 0:
                raise ValueError(f"Invalid GPU-memory metadata for {run_key}/h{horizon}")
            if not metadata.get("gpu_name") or not metadata.get("torch_version") or not metadata.get("cuda_version"):
                raise ValueError(f"Missing software or GPU metadata for {run_key}/h{horizon}")
            if metadata.get("deterministic_algorithms") is not True:
                raise ValueError(f"Deterministic-algorithm controls were not enabled for {run_key}/h{horizon}")
            path_metrics = metrics(truth, prediction, scales[task])
            terminal_metrics = metrics(truth[:, -1:, :], prediction[:, -1:, :], scales[task])
            saved = np.load(directory / "metrics.npy")
            for index, metric_name in ((0, "mae"), (1, "mse"), (2, "rmse"), (5, "rmsse")):
                if not np.isclose(float(saved[index]), path_metrics[metric_name], rtol=1e-6, atol=1e-10):
                    raise ValueError(f"Saved {metric_name.upper()} mismatch for {run_key}/h{horizon}")
            rows.append(
                {
                    "dataset": task,
                    "model": model,
                    "seed": seed,
                    "horizon": horizon,
                    "origin_count": len(origin_dates),
                    "runtime_seconds": runtime_seconds,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                    **{f"path_{key}": value for key, value in path_metrics.items()},
                    **{f"terminal_{key}": value for key, value in terminal_metrics.items()},
                    "setting": setting,
                }
            )
            loss = np.mean((truth - prediction) ** 2, axis=(1, 2))
            losses[(task, model, horizon, seed)] = loss
            origins[(task, model, horizon, seed)] = origin_dates
    return pd.DataFrame(rows), {"losses": losses, "origins": origins}


def load_statistical(protocol: dict, manifest: dict, scales: dict) -> tuple[pd.DataFrame, dict]:
    require_artifact(STATISTICAL_METRICS, "statistical-baseline metrics")
    table = pd.read_csv(STATISTICAL_METRICS)
    expected = {
        (task, model, int(horizon))
        for task, config in manifest.items()
        for model in (*STATISTICAL_COMMON, *(("VAR-difference",) if config["features"] == "MS" else ()))
        for horizon in protocol["horizons"]
    }
    observed = {
        (str(row.dataset), str(row.model), int(row.horizon))
        for row in table.itertuples()
    }
    if observed != expected or len(table) != len(expected):
        raise RuntimeError(
            f"Incomplete statistical matrix: rows={len(table)}, expected={len(expected)}, "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    origins_expected = expected_origins(manifest, protocol)
    rows = []
    losses = {}
    origins = {}
    for record in table.to_dict("records"):
        task = record["dataset"]
        horizon = int(record["horizon"])
        bundle = np.load(REPO_ROOT / record["bundle"], allow_pickle=False)
        prediction = bundle["prediction"]
        truth = bundle["truth"]
        origin_dates = bundle["origin_dates"].astype(str)
        metadata_path = (REPO_ROOT / record["bundle"]).with_name("meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "protocol_version": protocol["protocol_version"],
            "dataset": task,
            "data_sha256": manifest[task]["data_sha256"],
            "model": record["model"],
            "horizon": horizon,
            "evaluation_split": "test",
        }
        for field, expected_value in expected_metadata.items():
            if metadata.get(field) != expected_value:
                raise ValueError(
                    f"Statistical metadata mismatch for {task}/{record['model']}/h{horizon}: {field}"
                )
        if not metadata.get("statsmodels_version") or not metadata.get("python_version"):
            raise ValueError(f"Missing statistical software metadata for {task}/{record['model']}/h{horizon}")
        if record["model"] == "ARIMA":
            selection = metadata.get("arima_selection", {})
            if (
                metadata.get("coefficient_fit_converged") is not True
                or selection.get("selected_converged") is not True
                or int(selection.get("converged_candidate_count", 0)) <= 0
            ):
                raise ValueError(f"Missing converged ARIMA audit for {task}/h{horizon}")
        if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != horizon:
            raise ValueError(f"Statistical prediction shape mismatch for {task}/{record['model']}/h{horizon}")
        if not np.array_equal(origin_dates, origins_expected[(task, horizon)]):
            raise ValueError(f"Statistical test origins mismatch for {task}/{record['model']}/h{horizon}")
        path_metrics = metrics(truth, prediction, scales[task])
        terminal_metrics = metrics(truth[:, -1:], prediction[:, -1:], scales[task])
        for metric_name in ("mae", "mse", "rmse", "rmsse"):
            if not np.isclose(
                float(record[metric_name]), path_metrics[metric_name], rtol=1e-6, atol=1e-10
            ):
                raise ValueError(
                    f"Statistical {metric_name.upper()} mismatch for "
                    f"{task}/{record['model']}/h{horizon}"
                )
        rows.append(
            {
                "dataset": task,
                "model": record["model"],
                "seed": "deterministic",
                "horizon": horizon,
                "origin_count": len(origin_dates),
                **{f"path_{key}": value for key, value in path_metrics.items()},
                **{f"terminal_{key}": value for key, value in terminal_metrics.items()},
                "setting": record["bundle"],
            }
        )
        loss = np.mean((truth - prediction) ** 2, axis=1)
        losses[(task, record["model"], horizon, "deterministic")] = loss
        origins[(task, record["model"], horizon, "deterministic")] = origin_dates
    return pd.DataFrame(rows), {"losses": losses, "origins": origins}


def aggregate_metrics(neural: pd.DataFrame, statistical: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "path_mae", "path_mse", "path_rmse", "path_rmsse",
        "terminal_mae", "terminal_mse", "terminal_rmse", "terminal_rmsse",
    ]
    neural_agg = neural.groupby(["dataset", "model", "horizon"], as_index=False)[value_columns].agg(["mean", "std"])
    neural_agg.columns = [
        "_".join(filter(None, map(str, column))).rstrip("_")
        for column in neural_agg.columns.to_flat_index()
    ]
    deterministic = statistical[["dataset", "model", "horizon", *value_columns]].copy()
    for column in value_columns:
        deterministic[f"{column}_mean"] = deterministic.pop(column)
        deterministic[f"{column}_std"] = 0.0
    return pd.concat([neural_agg, deterministic], ignore_index=True, sort=False)


def add_skill(table: pd.DataFrame) -> pd.DataFrame:
    naive = table[table["model"] == "Last-value"].set_index(["dataset", "horizon"])["path_mse_mean"]
    output = table.copy()
    output["skill_vs_last_value"] = [
        1.0 - row.path_mse_mean / naive.loc[(row.dataset, row.horizon)]
        for row in output.itertuples()
    ]
    return output


def s_ms_comparison(table: pd.DataFrame) -> pd.DataFrame:
    paired = table.copy()
    paired["mode"] = paired["dataset"].str.extract(r"_(s|ms)$", expand=False)
    if paired["mode"].isna().any():
        raise ValueError("Financial task IDs must end in _s or _ms")
    paired["family"] = paired["dataset"].str.replace(r"_(s|ms)$", "", regex=True)
    values = paired[
        ["family", "mode", "dataset", "model", "horizon", "path_rmsse_mean", "skill_vs_last_value"]
    ]
    s_rows = values[values["mode"] == "s"].rename(
        columns={
            "dataset": "s_task",
            "path_rmsse_mean": "s_path_rmsse",
            "skill_vs_last_value": "s_skill_vs_last_value",
        }
    )
    ms_rows = values[values["mode"] == "ms"].rename(
        columns={
            "dataset": "ms_task",
            "path_rmsse_mean": "ms_path_rmsse",
            "skill_vs_last_value": "ms_skill_vs_last_value",
        }
    )
    output = s_rows.drop(columns="mode").merge(
        ms_rows.drop(columns="mode"), on=["family", "model", "horizon"], how="inner", validate="one_to_one"
    )
    output["ms_minus_s_path_rmsse"] = output["ms_path_rmsse"] - output["s_path_rmsse"]
    output["ms_minus_s_skill"] = output["ms_skill_vs_last_value"] - output["s_skill_vs_last_value"]
    return output.sort_values(["family", "model", "horizon"]).reset_index(drop=True)


def regime_by_date(manifest: dict, protocol: dict) -> dict[str, dict[str, str]]:
    regime = protocol["regime_analysis"]
    window = int(regime["lookback"])
    lag = int(regime["availability_lag"])
    low_quantile, high_quantile = [float(value) for value in regime["training_quantiles"]]
    low_label, middle_label, high_label = regime["labels"]
    output = {}
    for task, config in manifest.items():
        frame = pd.read_csv(REPO_ROOT / config["root_path"] / config["data_path"], usecols=["date", config["target"]])
        dates = pd.to_datetime(frame["date"], errors="raise")
        volatility = frame[config["target"]].astype(float).diff().rolling(window).std().shift(lag)
        train_values = volatility[dates <= pd.Timestamp(protocol["splits"]["train_end"])].dropna()
        low_threshold = float(train_values.quantile(low_quantile))
        high_threshold = float(train_values.quantile(high_quantile))
        labels = np.where(
            volatility <= low_threshold,
            low_label,
            np.where(volatility >= high_threshold, high_label, middle_label),
        )
        output[task] = dict(zip(dates.dt.strftime("%Y-%m-%d"), labels))
    return output


def regime_metrics(protocol: dict, manifest: dict, *stores: dict) -> pd.DataFrame:
    assignments = regime_by_date(manifest, protocol)
    labels = protocol["regime_analysis"]["labels"]
    rows = []
    for store in stores:
        for (task, model, horizon, seed), losses in store["losses"].items():
            origins = store["origins"][(task, model, horizon, seed)]
            origin_regimes = np.asarray([assignments[task][str(date)] for date in origins])
            for label in labels:
                selected = losses[origin_regimes == label]
                rows.append(
                    {
                        "dataset": task,
                        "model": model,
                        "horizon": int(horizon),
                        "seed": seed,
                        "regime": label,
                        "origin_count": int(len(selected)),
                        "origin_mse": float(np.mean(selected)) if len(selected) else float("nan"),
                    }
                )
    detailed = pd.DataFrame(rows)
    neural = detailed[detailed["seed"] != "deterministic"]
    deterministic = detailed[detailed["seed"] == "deterministic"]
    neural_aggregate = neural.groupby(
        ["dataset", "model", "horizon", "regime"], as_index=False
    ).agg(
        origin_mse_mean=("origin_mse", "mean"),
        origin_mse_std=("origin_mse", "std"),
        origin_count=("origin_count", "min"),
    )
    deterministic_aggregate = deterministic[
        ["dataset", "model", "horizon", "regime", "origin_count", "origin_mse"]
    ].rename(columns={"origin_mse": "origin_mse_mean"})
    deterministic_aggregate["origin_mse_std"] = 0.0
    aggregate = pd.concat([neural_aggregate, deterministic_aggregate], ignore_index=True, sort=False)
    naive = aggregate[aggregate["model"] == "Last-value"].set_index(
        ["dataset", "horizon", "regime"]
    )["origin_mse_mean"]
    aggregate["skill_vs_last_value"] = [
        1.0 - row.origin_mse_mean / naive.loc[(row.dataset, row.horizon, row.regime)]
        for row in aggregate.itertuples()
    ]
    return aggregate.sort_values(["dataset", "model", "horizon", "regime"]).reset_index(drop=True)


def moving_block_bootstrap(diff: np.ndarray, horizon: int, replicates: int, rng: np.random.Generator) -> dict:
    n = len(diff)
    block = min(n, max(int(horizon), int(math.ceil(n ** (1 / 3)))))
    blocks = int(math.ceil(n / block))
    means = np.empty(replicates, dtype=float)
    base = np.arange(block)
    for index in range(replicates):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + base[None, :]) % n).reshape(-1)[:n]
        means[index] = float(np.mean(diff[indices]))
    observed = float(np.mean(diff))
    p_value = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {
        "mean_loss_difference": observed,
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_value": p_value,
        "block_length": block,
        "origin_count": n,
    }


def best_validation_baselines(protocol: dict, tasks: list[str]) -> dict[str, str]:
    selected = {}
    for task in tasks:
        path = REPO_ROOT / "results" / "optuna" / task / "baseline_best_params.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = {
            model: payload["models"][model]["best_validation_replay_balanced_rmsse"]
            for model in NON_MAMBA
        }
        selected[task] = min(candidates, key=candidates.get)
    return selected


def mean_seed_loss(store: dict, task: str, model: str, horizon: int, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [store["losses"][(task, model, horizon, seed)] for seed in seeds]
    date_arrays = [store["origins"][(task, model, horizon, seed)] for seed in seeds]
    if any(not np.array_equal(date_arrays[0], value) for value in date_arrays[1:]):
        raise ValueError(f"Seed origins differ for {task}/{model}/h{horizon}")
    return np.mean(np.stack(arrays), axis=0), date_arrays[0]


def bootstrap_contrasts(protocol: dict, neural_store: dict, replicates: int, seed: int) -> pd.DataFrame:
    tasks = sorted({key[0] for key in neural_store["losses"]})
    best_baseline = best_validation_baselines(protocol, tasks)
    rng = np.random.default_rng(seed)
    rows = []
    for task in tasks:
        for proposal in MAMBA_PROPOSALS:
            for comparator in ("InvertedMamba", best_baseline[task]):
                for horizon in protocol["horizons"]:
                    proposal_loss, proposal_dates = mean_seed_loss(
                        neural_store, task, proposal, horizon, protocol["final_seeds"]
                    )
                    comparator_loss, comparator_dates = mean_seed_loss(
                        neural_store, task, comparator, horizon, protocol["final_seeds"]
                    )
                    if not np.array_equal(proposal_dates, comparator_dates):
                        raise ValueError(f"Contrast origins differ for {task}/h{horizon}")
                    result = moving_block_bootstrap(
                        proposal_loss - comparator_loss,
                        horizon,
                        replicates,
                        rng,
                    )
                    rows.append(
                        {
                            "dataset": task,
                            "proposal": proposal,
                            "comparator": comparator,
                            "horizon": horizon,
                            **result,
                        }
                    )
    output = pd.DataFrame(rows)
    output["holm_p_value"] = np.nan
    for _, indices in output.groupby(["dataset", "proposal", "comparator"]).groups.items():
        ordered = sorted(indices, key=lambda index: output.loc[index, "p_value"])
        running = 0.0
        count = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(output.loc[index, "p_value"]))
            running = max(running, adjusted)
            output.loc[index, "holm_p_value"] = running
    return output


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scales = target_scales(manifest, protocol)
    neural, neural_store = load_neural(protocol, manifest)
    statistical, statistical_store = load_statistical(protocol, manifest, scales)
    aggregate = add_skill(aggregate_metrics(neural, statistical))
    mode_comparison = s_ms_comparison(aggregate)
    regimes = regime_metrics(protocol, manifest, neural_store, statistical_store)
    contrasts = bootstrap_contrasts(
        protocol,
        neural_store,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    neural.to_csv(ANALYSIS_ROOT / "neural_metrics_by_seed.csv", index=False)
    statistical.to_csv(ANALYSIS_ROOT / "statistical_metrics.csv", index=False)
    aggregate.to_csv(ANALYSIS_ROOT / "aggregate_metrics.csv", index=False)
    mode_comparison.to_csv(ANALYSIS_ROOT / "s_ms_comparison.csv", index=False)
    regimes.to_csv(ANALYSIS_ROOT / "regime_metrics.csv", index=False)
    contrasts.to_csv(ANALYSIS_ROOT / "bootstrap_contrasts.csv", index=False)
    atomic_write_json(
        ANALYSIS_ROOT / "analysis_manifest.json",
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": next(iter({config["protocol_sha256"] for config in manifest.values()})),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "neural_run_count": int(len(neural)),
            "statistical_run_count": int(len(statistical)),
            "contrast_count": int(len(contrasts)),
            "s_ms_pair_count": int(len(mode_comparison)),
            "regime_metric_count": int(len(regimes)),
        },
    )
    print(f"wrote analysis artifacts to {ANALYSIS_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Analysis blocked: {exc}") from None
