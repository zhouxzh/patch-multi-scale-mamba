#!/usr/bin/env python
"""Train the active S-only financial model matrix with retained artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.financial.evaluation import build_base_args, evaluate_horizons, set_global_seed
from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.registry import public_name, runtime_model_name
from utils.optuna_support import load_dataset_manifest


CONFIG_PATH = REPO_ROOT / "configs" / "financial_s_only_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "financial_s_only_retrospective_v1"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "financial_s_only_retrospective_v1"
SUMMARY_NAME = "campaign_summary.json"
PLAN_NAME = "campaign_plan.json"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
ARTIFACT_MANIFEST = "artifact_manifest.json"
RESULT_FILES = (
    "pred.npy",
    "true.npy",
    "metrics.npy",
    "origin_dates.npy",
    "forecast_dates.npy",
    "meta.json",
)
REPLAY_VALIDATION_NAME = "replay_validation.json"
REPLAY_COMPARISON_ATOL = 1e-6


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol_version", "base_protocol", "selection_snapshot", "training", "models"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Mamba rerun configuration is missing fields: {missing}")
    return payload


def select(requested: list[str], available: list[str], label: str) -> list[str]:
    if requested == ["ALL"]:
        return list(available)
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; available={available}")
    return list(requested)


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mamba rerun campaign")


def resolve_scope(config: dict, args: argparse.Namespace) -> tuple[list[str], list[str], list[int], list[int], dict]:
    manifest_path = REPO_ROOT / config["base_protocol"]["training_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = select(args.tasks, sorted(manifest), "task IDs")
    invalid_tasks = [task for task in tasks if not task.endswith("_s")]
    if invalid_tasks:
        raise ValueError(
            "The active financial S-only workflow accepts only target-only tasks: "
            + ", ".join(invalid_tasks)
        )
    if any(entry.get("features") != "S" for task, entry in manifest.items() if task in tasks):
        raise ValueError("S-only manifest contains a non-S task")
    models = select(args.models, list(config["models"]), "model IDs")
    allowed_horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    requested_horizons = [str(value) for value in (args.horizons or allowed_horizons)]
    horizons = [int(value) for value in select(requested_horizons, [str(value) for value in allowed_horizons], "horizons")]
    allowed_seeds = [int(value) for value in config["base_protocol"]["seeds"]]
    seeds = [int(value) for value in select([str(value) for value in args.seeds], [str(value) for value in allowed_seeds], "seeds")]
    return tasks, models, horizons, seeds, manifest


def selection_params(config: dict, task: str, model_id: str) -> tuple[dict, Path]:
    model = config["models"][model_id]
    selection_file = model.get("selection_file", config.get("selection_file", "mamba_best_params.json"))
    selection_task = task
    task_map = config.get("selection_task_map", {})
    if task_map:
        for family, mapped_task in task_map.items():
            if task.startswith(f"financial_{family}_"):
                selection_task = mapped_task
                break
    selection_root = model.get("selection_snapshot", config.get("selection_snapshot"))
    path = REPO_ROOT / selection_root / selection_task / selection_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing selection snapshot: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") not in {task, selection_task}:
        raise ValueError(f"Selection snapshot dataset mismatch: {path}")
    if not str(payload.get("search_space_version", "")).startswith(config["selection_search_space_prefix"]):
        raise ValueError(f"Selection snapshot version mismatch: {path}")
    selection = payload.get("models", {}).get(model["selection_model"], {})
    if selection.get("selection_eval_flag") != "val" or not selection.get("params"):
        raise ValueError(f"Selection snapshot is incomplete for {task}/{model_id}: {path}")
    params = dict(selection["params"])
    params.update(model.get("parameter_overrides", {}))
    params.update(
        loss=config["training"]["loss"],
        horizon_weighting=config["training"]["horizon_weighting"],
    )
    return params, path


def stage_tag(model_id: str, seed: int) -> str:
    short_names = {
        "RegimeGatedMamba": "rgm",
        "UngatedRegimeLinearMamba": "urgm",
        "MultiScalePatchMamba": "msp",
        "InvertedMamba": "inv",
        "DecompositionMamba": "dec",
        "MultiScaleDecompositionMamba": "msd",
        "DLinear": "dlinear",
        "NLinear": "nlinear",
        "PatchTST": "patchtst",
        "TimesNet": "timesnet",
        "iTransformer": "itransformer",
    }
    return f"rerun_{short_names[model_id]}_s{seed}"


def storage_model_label(model_id: str) -> str:
    """Return a compact, deterministic directory token for retained artifacts."""
    return {
        "RegimeGatedMamba": "rgm",
        "UngatedRegimeLinearMamba": "urgm",
        "MultiScalePatchMamba": "msp",
        "InvertedMamba": "inv",
        "DecompositionMamba": "dec",
        "MultiScaleDecompositionMamba": "msd",
        "DLinear": "dlinear",
        "NLinear": "nlinear",
        "PatchTST": "patchtst",
        "TimesNet": "timesnet",
        "iTransformer": "itransformer",
    }[model_id]


def runner_args(args: argparse.Namespace, config: dict, seed: int) -> Namespace:
    training = config["training"]
    return Namespace(
        phase="mamba_rerun",
        seed=int(seed),
        cpu=False,
        require_cuda=True,
        gpu=int(args.gpu),
        train_epochs=int(args.train_epochs or training["train_epochs"]),
        patience=int(args.patience or training["patience"]),
        num_workers=int(args.num_workers if args.num_workers is not None else training["num_workers"]),
        tensorboard_dir=str(args.output_root / "tensorboard"),
        checkpoints=str(args.checkpoint_root),
        results_root=str(args.output_root / "runs"),
        no_save_final_artifacts=False,
        discard_final_checkpoints=False,
    )


def summary_path(output_root: Path) -> Path:
    return output_root / SUMMARY_NAME


def load_summary(output_root: Path, config: dict) -> dict:
    path = summary_path(output_root)
    if not path.is_file():
        return {"protocol_version": config["protocol_version"], "runs": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != config["protocol_version"]:
        raise ValueError(f"Campaign summary has a different protocol: {path}")
    return payload


def artifact_paths(output_root: Path, checkpoint_root: Path, setting: str) -> dict[str, Path]:
    run_root = output_root / "runs" / setting
    return {
        "checkpoint.pth": checkpoint_root / setting / "checkpoint.pth",
        **{name: run_root / name for name in RESULT_FILES},
    }


def hashed_artifacts(output_root: Path, checkpoint_root: Path, settings: dict) -> dict[str, dict]:
    artifacts = {}
    for horizon, setting in sorted(settings.items(), key=lambda item: int(item[0])):
        paths = artifact_paths(output_root, checkpoint_root, str(setting))
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing saved artifacts for {setting}: {missing}")
        artifacts[str(horizon)] = {
            "setting": str(setting),
            "files": {
                name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
                for name, path in paths.items()
            },
        }
    return artifacts


def record_is_complete(record: dict, horizons: list[int], output_root: Path, checkpoint_root: Path) -> bool:
    if record.get("status") != "complete":
        return False
    artifacts = record.get("artifacts", {})
    if set(artifacts) != {str(value) for value in horizons}:
        return False
    for horizon in horizons:
        artifact = artifacts[str(horizon)]
        setting = artifact.get("setting")
        if not setting:
            return False
        for name, expected in artifact.get("files", {}).items():
            path = artifact_paths(output_root, checkpoint_root, setting).get(name)
            if path is None or not path.is_file() or sha256_path(path) != expected.get("sha256"):
                return False
    return True


def campaign_plan(config: dict, tasks: list[str], models: list[str], horizons: list[int], seeds: list[int], manifest: dict) -> dict:
    tuples = [
        {"task": task, "model_id": model, "seed": seed, "horizons": horizons}
        for task in tasks
        for model in models
        for seed in seeds
    ]
    return {
        "protocol_version": config["protocol_version"],
        "evidence_status": config["evidence_status"],
        "task_count": len(tasks),
        "model_count": len(models),
        "seed_count": len(seeds),
        "horizon_count": len(horizons),
        "planned_horizon_runs": len(tuples) * len(horizons),
        "runs": tuples,
        "data_sha256": {task: manifest[task]["data_sha256"] for task in tasks},
    }


def enrich_metadata(output_root: Path, settings: dict, config: dict, task: str, model_id: str, selection_path: Path) -> None:
    model = config["models"][model_id]
    for setting in settings.values():
        path = output_root / "runs" / setting / "meta.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(
            campaign_protocol_version=config["protocol_version"],
            campaign_evidence_status=config["evidence_status"],
            canonical_model_id=model_id,
            source_model_id=model["source_model_id"],
            public_name=public_name(model_id),
            model_role=model["role"],
            selection_source=str(selection_path.relative_to(REPO_ROOT)),
            selection_source_sha256=sha256_path(selection_path),
            checkpoint_retained=True,
            forecast_data_retained=True,
        )
        atomic_write_json(path, metadata)


def build_manifests(output_root: Path, checkpoint_root: Path, summary: dict) -> dict:
    checkpoint_entries, artifact_entries, missing = [], [], []
    for run_key, record in sorted(summary.get("runs", {}).items()):
        for horizon, artifact in sorted(record.get("artifacts", {}).items(), key=lambda item: int(item[0])):
            setting = artifact.get("setting", "")
            files = artifact.get("files", {})
            checkpoint = files.get("checkpoint.pth")
            if not checkpoint:
                missing.append(f"{run_key}|h{horizon}:checkpoint")
                continue
            common = {
                "run_key": run_key,
                "dataset": record.get("dataset"),
                "model_id": record.get("model_id"),
                "seed": record.get("seed"),
                "horizon": int(horizon),
                "setting": setting,
            }
            checkpoint_entries.append({
                **common,
                "checkpoint": str((checkpoint_root / setting / "checkpoint.pth").relative_to(REPO_ROOT)),
                **checkpoint,
            })
            artifact_entries.append({**common, "files": files})
    checkpoint_payload = {
        "manifest_version": "financial_s_only_checkpoint_v1",
        "protocol_version": summary.get("protocol_version"),
        "entries": checkpoint_entries,
        "missing": missing,
    }
    artifact_payload = {
        "manifest_version": "financial_s_only_artifact_v1",
        "protocol_version": summary.get("protocol_version"),
        "entries": artifact_entries,
        "missing": missing,
    }
    atomic_write_json(checkpoint_root / CHECKPOINT_MANIFEST, checkpoint_payload)
    atomic_write_json(output_root / ARTIFACT_MANIFEST, artifact_payload)
    return {"checkpoint_entries": len(checkpoint_entries), "missing": len(missing)}


def train(args: argparse.Namespace) -> dict:
    if not getattr(args, "unlock_retrospective_test", False):
        raise RuntimeError(
            "Retrospective training writes a new test-result release; pass "
            "--unlock-retrospective-test after reviewing the frozen protocol."
        )
    config = load_config(args.config)
    tasks, models, horizons, seeds, manifest = resolve_scope(config, args)
    require_cuda()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / PLAN_NAME, campaign_plan(config, tasks, models, horizons, seeds, manifest))
    summary = load_summary(args.output_root, config)
    summary.update(
        protocol_version=config["protocol_version"],
        evidence_status=config["evidence_status"],
        tasks=tasks,
        models=models,
        horizons=horizons,
        seeds=seeds,
    )
    completed_now = 0
    for task in tasks:
        dataset_cfg = load_dataset_manifest(REPO_ROOT / config["base_protocol"]["training_manifest"], task, REPO_ROOT)
        for model_id in models:
            model = config["models"][model_id]
            params, selection_path = selection_params(config, task, model_id)
            for seed in seeds:
                key = f"{task}|{model_id}|{seed}"
                if record_is_complete(summary["runs"].get(key, {}), horizons, args.output_root, args.checkpoint_root):
                    print(f"skip complete {key}", flush=True)
                    continue
                if args.max_runs and completed_now >= args.max_runs:
                    atomic_write_json(summary_path(args.output_root), summary)
                    manifests = build_manifests(args.output_root, args.checkpoint_root, summary)
                    return {"status": "partial", "completed_horizon_runs": completed_now, **manifests}
                cli = runner_args(args, config, seed)
                cli.storage_model_label = storage_model_label(model_id)
                metrics, aggregate, settings = evaluate_horizons(
                    cli,
                    dataset_cfg,
                    runtime_model_name(model_id),
                    params,
                    horizons,
                    stage_tag(model_id, seed),
                    save_artifacts=True,
                    eval_flag="test",
                )
                enrich_metadata(args.output_root, settings, config, task, model_id, selection_path)
                artifacts = hashed_artifacts(args.output_root, args.checkpoint_root, settings)
                summary["runs"][key] = {
                    "status": "complete",
                    "dataset": task,
                    "model_id": model_id,
                    "source_model_id": model["source_model_id"],
                    "public_name": public_name(model_id),
                    "runtime_model": runtime_model_name(model_id),
                    "seed": seed,
                    "params": params,
                    "selection_source": str(selection_path.relative_to(REPO_ROOT)),
                    "selection_source_sha256": sha256_path(selection_path),
                    "per_horizon_metrics": metrics,
                    "aggregate": aggregate,
                    "settings": {str(horizon): setting for horizon, setting in settings.items()},
                    "artifacts": artifacts,
                }
                atomic_write_json(summary_path(args.output_root), summary)
                completed_now += len(horizons)
    manifests = build_manifests(args.output_root, args.checkpoint_root, summary)
    return {"status": "complete", "completed_horizon_runs": completed_now, **manifests}


def replay(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    tasks, models, horizons, seeds, _ = resolve_scope(config, args)
    require_cuda()
    summary = load_summary(args.output_root, config)
    args.replay_output_root.mkdir(parents=True, exist_ok=True)
    from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast

    rows = []
    for task in tasks:
        dataset_cfg = load_dataset_manifest(REPO_ROOT / config["base_protocol"]["training_manifest"], task, REPO_ROOT)
        for model_id in models:
            for seed in seeds:
                key = f"{task}|{model_id}|{seed}"
                record = summary.get("runs", {}).get(key, {})
                if not record_is_complete(record, horizons, args.output_root, args.checkpoint_root):
                    raise RuntimeError(f"Cannot replay incomplete run: {key}")
                for horizon in horizons:
                    setting = record["settings"][str(horizon)]
                    cli = runner_args(args, config, seed)
                    cli.storage_model_label = storage_model_label(model_id)
                    cli.results_root = str(args.replay_output_root / "runs")
                    model_args = build_base_args(
                        cli,
                        dataset_cfg,
                        runtime_model_name(model_id),
                        record["params"],
                        horizon,
                        f"replay_{stage_tag(model_id, seed)}_h{horizon}",
                    )
                    set_global_seed(seed)
                    exp = Exp_Long_Term_Forecast(model_args)
                    try:
                        evaluation = exp.evaluate(
                            setting=setting,
                            load_checkpoint=True,
                            save_artifacts=True,
                            save_visuals=False,
                            write_result_log=False,
                            write_tensorboard=False,
                            flag="test",
                        )
                    finally:
                        if getattr(exp, "writer", None) is not None:
                            exp.writer.close()
                    comparison = compare_replay_artifacts(
                        args.output_root,
                        args.replay_output_root,
                        setting,
                    )
                    if not comparison["valid"]:
                        raise RuntimeError(f"Checkpoint replay mismatch for {setting}: {comparison['problems']}")
                    rows.append({
                        "dataset": task,
                        "model_id": model_id,
                        "seed": seed,
                        "horizon": horizon,
                        "mae": float(evaluation["mae"]),
                        "mse": float(evaluation["mse"]),
                        "rmse": float(evaluation["rmse"]),
                        "rmsse": float(evaluation["rmsse"]),
                        "prediction_max_abs_difference": comparison["prediction_max_abs_difference"],
                        "truth_max_abs_difference": comparison["truth_max_abs_difference"],
                    })
    atomic_write_json(args.replay_output_root / "replay_summary.json", {"entries": rows})
    return {"status": "replayed", "entries": len(rows)}


def rebuild_replay_summary(args: argparse.Namespace) -> dict:
    """Rebuild the local replay index from a completed equality-validation record."""
    config = load_config(args.config)
    tasks, models, horizons, seeds, _ = resolve_scope(config, args)
    path = args.replay_output_root / REPLAY_VALIDATION_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing replay validation record: {path}")
    validation = json.loads(path.read_text(encoding="utf-8"))
    expected = len(tasks) * len(models) * len(horizons) * len(seeds)
    entries = validation.get("entries", [])
    requested = {
        (task, model_id, seed, horizon)
        for task in tasks
        for model_id in models
        for seed in seeds
        for horizon in horizons
    }
    selected = [
        entry for entry in entries
        if (entry.get("dataset"), entry.get("model_id"), entry.get("seed"), entry.get("horizon")) in requested
    ]
    observed = {
        (entry.get("dataset"), entry.get("model_id"), entry.get("seed"), entry.get("horizon"))
        for entry in selected
    }
    if (
        validation.get("status") != "verified"
        or validation.get("problems")
        or len(selected) != expected
        or observed != requested
        or any(not entry.get("valid") for entry in selected)
    ):
        raise RuntimeError("Cannot rebuild replay summary from an incomplete validation record")
    payload = {
        "status": "verified",
        "source": REPLAY_VALIDATION_NAME,
        "entries": sorted(selected, key=lambda row: (row["dataset"], row["model_id"], row["seed"], row["horizon"])),
    }
    atomic_write_json(args.replay_output_root / "replay_summary.json", payload)
    return {"status": "reindexed", "entries": len(selected), "source": str(path)}


def compare_replay_artifacts(
    output_root: Path,
    replay_output_root: Path,
    setting: str,
    atol: float = REPLAY_COMPARISON_ATOL,
) -> dict:
    """Compare a replayed forecast bundle with its retained source bundle."""
    source_root = output_root / "runs" / setting
    replay_root = replay_output_root / "runs" / setting
    result = {
        "setting": setting,
        "prediction_max_abs_difference": None,
        "truth_max_abs_difference": None,
        "origin_dates_match": False,
        "forecast_dates_match": False,
        "valid": False,
        "problems": [],
    }
    numeric_files = (("pred.npy", "prediction_max_abs_difference"), ("true.npy", "truth_max_abs_difference"))
    for filename, field in numeric_files:
        source_path = source_root / filename
        replay_path = replay_root / filename
        if not source_path.is_file() or not replay_path.is_file():
            result["problems"].append(f"missing:{filename}")
            continue
        source = np.load(source_path, allow_pickle=False)
        replayed = np.load(replay_path, allow_pickle=False)
        if source.shape != replayed.shape:
            result["problems"].append(f"shape:{filename}")
            continue
        maximum = float(np.max(np.abs(np.asarray(source, dtype=float) - np.asarray(replayed, dtype=float))))
        result[field] = maximum
        if not np.allclose(source, replayed, rtol=0.0, atol=atol):
            result["problems"].append(f"value:{filename}")
    for filename, field in (("origin_dates.npy", "origin_dates_match"), ("forecast_dates.npy", "forecast_dates_match")):
        source_path = source_root / filename
        replay_path = replay_root / filename
        if not source_path.is_file() or not replay_path.is_file():
            result["problems"].append(f"missing:{filename}")
            continue
        result[field] = bool(np.array_equal(np.load(source_path, allow_pickle=False), np.load(replay_path, allow_pickle=False)))
        if not result[field]:
            result["problems"].append(f"value:{filename}")
    result["valid"] = not result["problems"]
    return result


def verify_replay(args: argparse.Namespace) -> dict:
    """Validate all existing replay bundles without loading models or using CUDA."""
    config = load_config(args.config)
    tasks, models, horizons, seeds, _ = resolve_scope(config, args)
    summary = load_summary(args.output_root, config)
    entries, problems = [], []
    for task in tasks:
        for model_id in models:
            for seed in seeds:
                key = f"{task}|{model_id}|{seed}"
                record = summary.get("runs", {}).get(key, {})
                if not record_is_complete(record, horizons, args.output_root, args.checkpoint_root):
                    problems.append(f"incomplete_source:{key}")
                    continue
                for horizon in horizons:
                    setting = record["settings"][str(horizon)]
                    comparison = compare_replay_artifacts(args.output_root, args.replay_output_root, setting)
                    comparison.update({"dataset": task, "model_id": model_id, "seed": seed, "horizon": horizon})
                    entries.append(comparison)
                    if not comparison["valid"]:
                        problems.append(f"mismatch:{task}|{model_id}|{seed}|h{horizon}")
    expected_entries = len(tasks) * len(models) * len(seeds) * len(horizons)
    result = {
        "status": "verified" if len(entries) == expected_entries and not problems else "incomplete",
        "expected_entries": expected_entries,
        "validated_entries": len(entries),
        "problems": problems,
        "atol": REPLAY_COMPARISON_ATOL,
        "entries": entries,
    }
    atomic_write_json(args.replay_output_root / REPLAY_VALIDATION_NAME, result)
    return result


def verify(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    tasks, models, horizons, seeds, _ = resolve_scope(config, args)
    summary = load_summary(args.output_root, config)
    expected = [f"{task}|{model}|{seed}" for task in tasks for model in models for seed in seeds]
    missing = [
        key for key in expected
        if not record_is_complete(summary.get("runs", {}).get(key, {}), horizons, args.output_root, args.checkpoint_root)
    ]
    manifests = build_manifests(args.output_root, args.checkpoint_root, summary)
    return {
        "status": "verified" if not missing and not manifests["missing"] else "incomplete",
        "expected_horizon_runs": len(expected) * len(horizons),
        "complete_run_bundles": len(expected) - len(missing),
        "missing_run_bundles": missing,
        **manifests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "replay", "replay-verify", "verify"], required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--replay-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "replay")
    parser.add_argument("--tasks", nargs="+", default=["ALL"])
    parser.add_argument("--models", nargs="+", default=["ALL"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025, 2026, 2027])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=0, help="Maximum newly completed horizon runs before stopping.")
    parser.add_argument(
        "--unlock-retrospective-test",
        action="store_true",
        help="Authorize creation of a new retrospective test-result release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.replay_output_root = args.replay_output_root.resolve()
    outcome = {"train": train, "replay": replay, "replay-verify": verify_replay, "verify": verify}[args.mode](args)
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Mamba rerun blocked: {exc}") from None
