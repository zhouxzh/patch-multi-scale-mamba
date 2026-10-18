#!/usr/bin/env python
"""Build and verify compact evidence tables for an S-only financial release."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.financial.mamba_rerun import artifact_paths, record_is_complete
from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.evidence import expected_origins, metrics, target_scales


DEFAULT_CONFIG = REPO_ROOT / "configs" / "financial_s_only_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "financial_s_only_retrospective_v1"
DEFAULT_CHECKPOINTS = REPO_ROOT / "checkpoints" / "financial_s_only_retrospective_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260812


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def release_paths(config: dict) -> tuple[dict, dict, Path]:
    protocol_path = REPO_ROOT / config["base_protocol"]["path"]
    manifest_path = REPO_ROOT / config["base_protocol"]["training_manifest"]
    return json.loads(protocol_path.read_text(encoding="utf-8")), json.loads(manifest_path.read_text(encoding="utf-8")), protocol_path


def expected_tasks(protocol: dict, manifest: dict) -> list[str]:
    tasks = list(protocol.get("tasks", protocol.get("target_roster", [])))
    if not tasks:
        tasks = sorted(manifest)
    if set(tasks) != set(manifest):
        raise ValueError("Release protocol task roster and training manifest differ")
    if any(not task.endswith("_s") or manifest[task].get("features") != "S" for task in tasks):
        raise ValueError("S-only release contains a non-S task")
    return sorted(tasks)


def _mean_scaled_loss(truth: np.ndarray, prediction: np.ndarray, scale: float) -> np.ndarray:
    """Return one training-scale-normalized squared loss per forecast origin."""
    return np.mean(np.square(truth - prediction), axis=(1, 2)) / float(scale)


def learned_rows(config: dict, protocol: dict, manifest: dict, output_root: Path, checkpoint_root: Path) -> tuple[list[dict], dict, list[str]]:
    summary_path = output_root / "campaign_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing learned campaign summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = expected_tasks(protocol, manifest)
    models = list(config["models"])
    horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    seeds = [int(value) for value in config["base_protocol"]["seeds"]]
    scales = target_scales({task: manifest[task] for task in tasks}, protocol)
    rows, problems, losses = [], [], {}
    for task in tasks:
        for model_id in models:
            for seed in seeds:
                key = f"{task}|{model_id}|{seed}"
                record = summary.get("runs", {}).get(key, {})
                if not record_is_complete(record, horizons, output_root, checkpoint_root):
                    problems.append(f"incomplete_learned:{key}")
                    continue
                for horizon in horizons:
                    setting = record["settings"][str(horizon)]
                    directory = output_root / "runs" / setting
                    prediction = np.load(directory / "pred.npy", mmap_mode="r")
                    truth = np.load(directory / "true.npy", mmap_mode="r")
                    origins = np.load(directory / "origin_dates.npy", allow_pickle=False).astype(str)
                    forecast_dates = np.load(directory / "forecast_dates.npy", allow_pickle=False).astype(str)
                    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
                    if prediction.shape != truth.shape or prediction.ndim != 3 or prediction.shape[1] != horizon:
                        problems.append(f"shape_learned:{key}:h{horizon}")
                        continue
                    if len(origins) != prediction.shape[0] or forecast_dates.shape != prediction.shape[:2]:
                        problems.append(f"dates_learned:{key}:h{horizon}")
                        continue
                    expected_origin_dates = expected_origins(manifest[task], protocol, horizon)
                    if not np.array_equal(origins, expected_origin_dates):
                        problems.append(f"origins_learned:{key}:h{horizon}")
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
                        problems.append(f"metadata_learned:{key}:h{horizon}")
                        continue
                    path_values = metrics(truth, prediction, scales[task])
                    terminal_values = metrics(truth[:, -1:, :], prediction[:, -1:, :], scales[task])
                    rows.append({
                        "dataset": task,
                        "model_id": model_id,
                        "model": record["public_name"],
                        "source_model_id": record["source_model_id"],
                        "seed": seed,
                        "horizon": horizon,
                        "origin_count": int(prediction.shape[0]),
                        "runtime_seconds": float(metadata["runtime_seconds"]),
                        "peak_gpu_memory_mb": float(metadata["peak_gpu_memory_mb"]),
                        **{f"path_{name}": value for name, value in path_values.items()},
                        **{f"terminal_{name}": value for name, value in terminal_values.items()},
                        "setting": setting,
                    })
                    losses[(task, model_id, horizon, seed)] = (
                        origins,
                        _mean_scaled_loss(truth, prediction, scales[task]),
                    )
    return rows, losses, problems


def deterministic_rows(config: dict, protocol: dict, manifest: dict, output_root: Path) -> tuple[list[dict], dict, list[str]]:
    path = output_root / "metrics" / "deterministic_metrics.csv"
    if not config["deterministic_models"]:
        return [], {}, []
    if not path.is_file():
        raise FileNotFoundError(f"Missing deterministic metrics: {path}")
    table = pd.read_csv(path)
    tasks = expected_tasks(protocol, manifest)
    models = list(config["deterministic_models"])
    horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    scales = target_scales({task: manifest[task] for task in tasks}, protocol)
    expected = {(task, model, horizon) for task in tasks for model in models for horizon in horizons}
    observed = {(str(row.dataset), str(row.model_id), int(row.horizon)) for row in table.itertuples()}
    problems = []
    if observed != expected:
        problems.append("deterministic_matrix_mismatch")
    losses = {}
    for row in table.to_dict("records"):
        bundle = Path(row["bundle"])
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        if not bundle.is_file():
            problems.append(f"missing_deterministic_bundle:{row['dataset']}:{row['model_id']}:h{row['horizon']}")
            continue
        payload = np.load(bundle, allow_pickle=False)
        truth = payload["truth"][:, :, None]
        prediction = payload["prediction"][:, :, None]
        origins = payload["origin_dates"].astype(str)
        expected_origin_dates = expected_origins(manifest[row["dataset"]], protocol, int(row["horizon"]))
        if not np.array_equal(origins, expected_origin_dates):
            problems.append(f"origins_deterministic:{row['dataset']}:{row['model_id']}:h{row['horizon']}")
            continue
        meta_path = bundle.with_name("meta.json")
        if not meta_path.is_file():
            problems.append(f"missing_deterministic_metadata:{row['dataset']}:{row['model_id']}:h{row['horizon']}")
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "protocol_version": config["protocol_version"],
            "dataset": row["dataset"],
            "model_id": row["model_id"],
            "horizon": int(row["horizon"]),
            "evaluation_split": "test",
            "data_sha256": manifest[row["dataset"]]["data_sha256"],
        }
        if any(metadata.get(name) != value for name, value in expected_metadata.items()):
            problems.append(f"metadata_deterministic:{row['dataset']}:{row['model_id']}:h{row['horizon']}")
            continue
        losses[(row["dataset"], row["model_id"], int(row["horizon"]), "deterministic")] = (
            origins,
            _mean_scaled_loss(truth, prediction, scales[row["dataset"]]),
        )
    return table.to_dict("records"), losses, problems


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in rows.columns if column.startswith(("path_", "terminal_", "runtime_", "peak_"))]
    learned = rows[rows["seed"] != "deterministic"]
    deterministic = rows[rows["seed"] == "deterministic"]
    learned_summary = learned.groupby(["dataset", "model_id", "model", "source_model_id", "horizon"], as_index=False)[numeric].agg(["mean", "std"])
    learned_summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in learned_summary.columns.to_flat_index()]
    learned_summary.insert(5, "seed_count", learned.groupby(["dataset", "model_id", "model", "source_model_id", "horizon"]).size().to_numpy())
    deterministic_summary = deterministic[["dataset", "model_id", "model", "source_model_id", "horizon", *numeric]].copy()
    deterministic_summary.insert(5, "seed_count", 0)
    for column in numeric:
        deterministic_summary[f"{column}_mean"] = deterministic_summary.pop(column)
        deterministic_summary[f"{column}_std"] = 0.0
    table = pd.concat([learned_summary, deterministic_summary], ignore_index=True, sort=False)
    table["path_rmsse_rank"] = table.groupby(["dataset", "horizon"])["path_rmsse_mean"].rank(method="min")
    table["path_rmse_rank"] = table.groupby(["dataset", "horizon"])["path_rmse_mean"].rank(method="min")
    table["path_mae_rank"] = table.groupby(["dataset", "horizon"])["path_mae_mean"].rank(method="min")
    return table.sort_values(["dataset", "horizon", "path_rmsse_rank", "model_id"]).reset_index(drop=True)


def contrasts(protocol: dict, learned_losses: dict, deterministic_losses: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    loss_series = {}
    tasks = sorted({key[0] for key in learned_losses})
    horizons = [int(value) for value in protocol["horizons"]]
    seeds = [int(value) for value in protocol["seeds"]]
    comparisons = tuple(protocol.get("contrast_references", ("LastValue", "RandomWalkDrift", "ARIMA", "UngatedRegimeLinearMamba")))
    for task in tasks:
        for model_id in ("RegimeGatedMamba",):
            for horizon in horizons:
                for reference in comparisons:
                    for seed in seeds:
                        origins, loss = learned_losses[(task, model_id, horizon, seed)]
                        if reference in {"LastValue", "RandomWalkDrift", "ARIMA"}:
                            ref_origins, ref_loss = deterministic_losses[(task, reference, horizon, "deterministic")]
                            reference_seed = "deterministic"
                        else:
                            ref_origins, ref_loss = learned_losses[(task, reference, horizon, seed)]
                            reference_seed = seed
                        if not np.array_equal(origins, ref_origins):
                            raise ValueError(f"Origin mismatch for {task}/{model_id}/{reference}/h{horizon}")
                        series_key = (task, model_id, reference, horizon, seed)
                        loss_series[series_key] = (loss, ref_loss)
                        difference = loss - ref_loss
                        rows.append({
                            "dataset": task,
                            "model_id": model_id,
                            "reference_model_id": reference,
                            "seed": seed,
                            "reference_seed": reference_seed,
                            "horizon": horizon,
                            "origin_count": len(difference),
                            "mean_path_scaled_mse_difference": float(np.mean(difference)),
                            "mean_path_rmsse_difference": float(np.sqrt(np.mean(loss)) - np.sqrt(np.mean(ref_loss))),
                            "win_fraction_vs_reference": float(np.mean(difference < 0.0)),
                        })
    return pd.DataFrame(rows), loss_series


def _moving_block_indices(count: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample paired forecast-origin indices from circular moving blocks."""
    if count == 0:
        raise ValueError("Cannot bootstrap an empty origin-level loss series")
    starts = rng.integers(0, count, size=int(np.ceil(count / block_length)))
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % count).reshape(-1)[:count]


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        running = max(running, (total - position) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def block_bootstrap(contrasts_table: pd.DataFrame, loss_series: dict) -> pd.DataFrame:
    """Nested task/seed and moving-origin-block bootstrap with Holm adjustment."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (model_id, reference, horizon), group in contrasts_table.groupby(["model_id", "reference_model_id", "horizon"]):
        task_to_seeds = {
            task: sorted(table["seed"].astype(int).tolist())
            for task, table in group.groupby("dataset")
        }
        tasks = sorted(task_to_seeds)
        block_length = max(1, int(horizon))
        samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for index in range(BOOTSTRAP_REPLICATES):
            task_means = []
            for task in rng.choice(tasks, size=len(tasks), replace=True):
                seed_values = []
                for seed in rng.choice(task_to_seeds[task], size=len(task_to_seeds[task]), replace=True):
                    model_loss, reference_loss = loss_series[(task, model_id, reference, int(horizon), int(seed))]
                    indices = _moving_block_indices(len(model_loss), block_length, rng)
                    seed_values.append(
                        float(
                            np.sqrt(np.mean(model_loss[indices]))
                            - np.sqrt(np.mean(reference_loss[indices]))
                        )
                    )
                task_means.append(float(np.mean(seed_values)))
            samples[index] = float(np.mean(task_means))
        observed = group["mean_path_rmsse_difference"].to_numpy(dtype=float)
        # Two-sided bootstrap sign probability is descriptive; Holm correction
        # covers the model/reference/horizon family in this release.
        p_value = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
        rows.append({
            "model_id": model_id,
            "reference_model_id": reference,
            "horizon": int(horizon),
            "task_cluster_count": int(len(tasks)),
            "task_seed_cluster_count": int(len(group)),
            "mean_path_rmsse_difference": float(np.mean(observed)),
            "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
            "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "origin_block_length": block_length,
            "p_value_descriptive": min(p_value, 1.0),
            "interpretation": "negative favors the named model; paired path-RMSSE differences are resampled over task, seed, and overlapping forecast-origin clusters",
        })
    table = pd.DataFrame(rows)
    table["p_value_holm"] = _holm_adjust(table["p_value_descriptive"].tolist())
    table["multiplicity_method"] = "Holm across all model-reference-horizon contrasts in this release"
    return table


def checksum_rows(root: Path, relative_paths: list[Path]) -> list[dict]:
    return [{"path": str(path.relative_to(root)), "sha256": sha256_path(path), "bytes": path.stat().st_size} for path in relative_paths if path.is_file()]


def checksum_problems(release_root: Path) -> list[str]:
    """Verify a completed release checksum manifest without trusting its presence."""
    path = release_root / "SHA256SUMS.json"
    if not path.is_file():
        return ["missing_release:SHA256SUMS.json"]
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["invalid_release:SHA256SUMS.json"]
    if not isinstance(rows, list) or not rows:
        return ["empty_release:SHA256SUMS.json"]
    problems = []
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str) or relative == "SHA256SUMS.json":
            problems.append("invalid_release:SHA256SUMS.json")
            continue
        artifact = release_root / relative
        if not artifact.is_file():
            problems.append(f"checksum_missing:{relative}")
        elif artifact.stat().st_size != row.get("bytes") or sha256_path(artifact) != row.get("sha256"):
            problems.append(f"checksum_mismatch:{relative}")
    return problems


def runtime_environment() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except subprocess.SubprocessError:
        commit = ""
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        ).splitlines()
    except subprocess.SubprocessError:
        status = []
    packages = {}
    for name in ("torch", "mamba-ssm", "causal-conv1d", "numpy", "pandas", "scipy", "statsmodels", "optuna"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {"available": False, "devices": []}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        cuda["error"] = str(exc)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_status_at_report": status,
        "packages": packages,
        "cuda": cuda,
    }


def build(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    protocol, manifest, protocol_path = release_paths(config)
    learned, learned_losses, learned_problems = learned_rows(config, protocol, manifest, args.output_root, args.checkpoint_root)
    deterministic, deterministic_losses, deterministic_problems = deterministic_rows(config, protocol, manifest, args.output_root)
    problems = learned_problems + deterministic_problems
    if problems:
        raise RuntimeError(f"Cannot build S-only report: {problems[:5]}")
    rows = pd.DataFrame([*learned, *deterministic])
    metrics_root = args.output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    rows.to_csv(metrics_root / "metrics_by_seed.csv", index=False)
    aggregate(rows).to_csv(metrics_root / "metrics_aggregate.csv", index=False)
    contrast_rows, loss_series = contrasts(protocol, learned_losses, deterministic_losses)
    contrast_rows.to_csv(metrics_root / "paired_contrasts.csv", index=False)
    block_bootstrap(contrast_rows, loss_series).to_csv(metrics_root / "paired_block_bootstrap.csv", index=False)
    release_root = REPO_ROOT / "reproducibility" / args.release
    release_metrics = release_root / "metrics"
    release_metrics.mkdir(parents=True, exist_ok=True)
    for name in ("metrics_by_seed.csv", "metrics_aggregate.csv", "paired_contrasts.csv", "paired_block_bootstrap.csv", "deterministic_metrics.csv"):
        source = metrics_root / name
        if source.is_file():
            (release_metrics / name).write_bytes(source.read_bytes())
    atomic_write_json(release_root / "runtime_environment.json", runtime_environment())
    profile_available = (release_metrics / "matched_cuda_microbenchmark.csv").is_file()
    allowed_claims = ["retrospective S-only path-RMSSE, MAE, and RMSE comparisons after verifier passes"]
    blocked_claims = ["untouched prospective-test performance"]
    if profile_available:
        allowed_claims.append("fixed-workload S-only CUDA inference comparison for profiled models")
    else:
        blocked_claims.append("speed comparison until a new S-only profile is supplied")
    claims = {
        "status": "profile_available" if profile_available else "pending_profile",
        "allowed_claims": allowed_claims,
        "blocked_claims": blocked_claims,
    }
    atomic_write_json(release_root / "claim_evidence_map.json", claims)
    # Make the final checksum file visible before checking release completeness.
    atomic_write_json(release_root / "SHA256SUMS.json", [])
    status = verify(args, write_status=True, validate_checksums=False)
    artifact_files = [path for path in release_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.json"]
    atomic_write_json(release_root / "SHA256SUMS.json", checksum_rows(release_root, artifact_files))
    # release_status.json is part of the manifest, so do not rewrite it after
    # checksums are recorded. The final pass only validates the frozen files.
    return verify(args, write_status=False)


def verify(
    args: argparse.Namespace,
    write_status: bool = False,
    validate_checksums: bool = True,
) -> dict:
    config = load_config(args.config)
    protocol, manifest, protocol_path = release_paths(config)
    tasks = expected_tasks(protocol, manifest)
    horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    seeds = [int(value) for value in config["base_protocol"]["seeds"]]
    expected_learned = len(tasks) * len(config["models"]) * len(seeds) * len(horizons)
    expected_deterministic = len(tasks) * len(config["deterministic_models"]) * len(horizons)
    summary_path = args.output_root / "campaign_summary.json"
    problems = []
    complete_learned = 0
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for task in tasks:
            for model in config["models"]:
                for seed in seeds:
                    if record_is_complete(summary.get("runs", {}).get(f"{task}|{model}|{seed}", {}), horizons, args.output_root, args.checkpoint_root):
                        complete_learned += len(horizons)
                    else:
                        problems.append(f"missing_learned:{task}|{model}|{seed}")
    else:
        problems.append("missing_campaign_summary")
    deterministic_path = args.output_root / "metrics" / "deterministic_metrics.csv"
    deterministic_table = pd.read_csv(deterministic_path) if deterministic_path.is_file() else pd.DataFrame()
    complete_deterministic = len(deterministic_table)
    expected_deterministic_cells = {
        (task, model, horizon)
        for task in tasks
        for model in config["deterministic_models"]
        for horizon in horizons
    }
    observed_deterministic_cells = {
        (str(row.dataset), str(row.model_id), int(row.horizon))
        for row in deterministic_table.itertuples()
    }
    if complete_deterministic != expected_deterministic or observed_deterministic_cells != expected_deterministic_cells:
        problems.append("deterministic_row_count")
    release_root = REPO_ROOT / "reproducibility" / args.release
    required = [
        "README.md",
        "protocol.json",
        "training_datasets.json",
        "model_identity_map.json",
        "selection_manifest.json",
        "config.json",
        "metrics/metrics_by_seed.csv",
        "metrics/metrics_aggregate.csv",
        "metrics/paired_contrasts.csv",
        "metrics/paired_block_bootstrap.csv",
        "runtime_environment.json",
        "claim_evidence_map.json",
        "SHA256SUMS.json",
    ]
    if config["deterministic_models"]:
        required.append("metrics/deterministic_metrics.csv")
    missing_release_files = [name for name in required if not (release_root / name).is_file()]
    if missing_release_files:
        problems.extend(f"missing_release:{name}" for name in missing_release_files)
    if validate_checksums and not missing_release_files and (release_root / "SHA256SUMS.json").is_file():
        problems.extend(checksum_problems(release_root))
    status = {
        "protocol_version": config["protocol_version"],
        "evidence_status": config["evidence_status"],
        "tasks": tasks,
        "expected_learned_horizon_rows": expected_learned,
        "complete_learned_horizon_rows": complete_learned,
        "expected_deterministic_rows": expected_deterministic,
        "complete_deterministic_rows": complete_deterministic,
        "status": "verified" if not problems else "incomplete",
        "problems": problems,
    }
    if write_status:
        atomic_write_json(release_root / "release_status.json", status)
    return status


def initialize_release_status(args: argparse.Namespace) -> dict:
    """Write live status before finalization; preserve a finalized checksum manifest."""
    release_root = REPO_ROOT / "reproducibility" / args.release
    # `release_status.json` is checksum-covered in a finalized release. A later
    # verify command must validate that immutable record rather than rewrite it.
    return verify(args, write_status=not (release_root / "SHA256SUMS.json").is_file())


def refresh_checksums(args: argparse.Namespace) -> dict:
    """Refresh checksums after an audited documentation-only release repair."""
    release_root = REPO_ROOT / "reproducibility" / args.release
    status = verify(args, write_status=False, validate_checksums=False)
    if status["status"] != "verified":
        raise RuntimeError(f"Cannot refresh checksums for an incomplete release: {status['problems']}")
    files = [path for path in release_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.json"]
    atomic_write_json(release_root / "SHA256SUMS.json", checksum_rows(release_root, files))
    return verify(args, write_status=False, validate_checksums=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "verify", "refresh-checksums"), required=True)
    parser.add_argument("--release", choices=("financial_s_only_v1", "financial_s_only_v2", "financial_s_only_v2_supplement"), default="financial_s_only_v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    print(json.dumps({"build": build, "verify": verify, "refresh-checksums": refresh_checksums}[args.mode](args), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"S-only report blocked: {exc}") from None
