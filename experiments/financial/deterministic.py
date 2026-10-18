#!/usr/bin/env python
"""Generate aligned deterministic references for an S-only financial release."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import atomic_write_json, sha256_path
from utils.financial.evidence import metrics, target_scales


DEFAULT_CONFIG = REPO_ROOT / "configs" / "financial_s_only_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "financial_s_only_retrospective_v1"
# The complete pre-specified low-order grid has 31 candidates: all p in 0..3,
# d in 0..1, q in 0..3, except the non-differenced white-noise model.
ARIMA_ORDERS = tuple(
    (p, d, q)
    for p in range(4)
    for d in range(2)
    for q in range(4)
    if (p, d, q) != (0, 0, 0)
)


def _select(values: list[str], available: list[str], label: str) -> list[str]:
    if values == ["ALL"]:
        return list(available)
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; available={available}")
    return list(values)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("deterministic_models"):
        raise ValueError("Release configuration has no deterministic models")
    return config


def _series_and_origins(entry: dict, protocol: dict, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(REPO_ROOT / entry["root_path"] / entry["data_path"])
    dates = pd.to_datetime(frame["date"], errors="raise")
    values = frame[entry["target"]].to_numpy(dtype=float)
    test_start = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_start"]), side="left"))
    test_end = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_end"]), side="right"))
    origin_indices = np.arange(test_start, test_end - int(horizon) + 1)
    if len(origin_indices) <= 0:
        raise ValueError(f"No test origins for {entry['data_path']} h={horizon}")
    origins = dates.iloc[origin_indices].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    truth = np.stack([values[index:index + int(horizon)] for index in origin_indices])
    return values, origin_indices, origins, truth


def last_value(values: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(values[origins - 1, None], int(horizon), axis=1)


def random_walk_drift(values: np.ndarray, origins: np.ndarray, horizon: int, history_start: int = 0) -> np.ndarray:
    output = []
    steps = np.arange(1, int(horizon) + 1, dtype=float)
    for origin in origins:
        count = max(1, int(origin) - history_start)
        drift = (values[int(origin) - 1] - values[history_start]) / count
        output.append(values[int(origin) - 1] + drift * steps)
    return np.asarray(output, dtype=float)


def _converged(fit: object) -> bool:
    return bool(
        fit.mle_retvals.get("converged", True)
        if hasattr(fit, "mle_retvals") else True
    )


def _fit_arima(values: np.ndarray, order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(values, order=order, trend="n").fit(method_kwargs={"maxiter": 500})


def _select_arima_order(training: np.ndarray, coefficient_data: np.ndarray) -> tuple[tuple[int, int, int], dict, object]:
    records = []
    for order in ARIMA_ORDERS:
        try:
            selection_fit = _fit_arima(training, order)
            coefficient_fit = _fit_arima(coefficient_data, order)
            records.append({
                "aic": float(selection_fit.aic),
                "order": order,
                "selection_converged": _converged(selection_fit),
                "coefficient_converged": _converged(coefficient_fit),
                "coefficient_fit": coefficient_fit,
            })
        except Exception:
            records.append({
                "aic": float("inf"),
                "order": order,
                "selection_converged": False,
                "coefficient_converged": False,
                "coefficient_fit": None,
            })
    eligible = [
        record for record in records
        if record["selection_converged"]
        and record["coefficient_converged"]
        and np.isfinite(record["aic"])
    ]
    if not eligible:
        raise RuntimeError("No ARIMA candidate converged in both pre-test selection and coefficient fitting")
    best = min(eligible, key=lambda record: record["aic"])
    return best["order"], {
        "candidate_orders": [list(order) for order in ARIMA_ORDERS],
        "attempted_candidate_count": len(records),
        "selection_converged_candidate_count": sum(record["selection_converged"] for record in records),
        "coefficient_converged_candidate_count": sum(record["coefficient_converged"] for record in records),
        "eligible_candidate_count": len(eligible),
        "selected_order": list(best["order"]),
        "selected_aic": float(best["aic"]),
        "selected_converged": True,
    }, best["coefficient_fit"]


def arima_forecasts(
    values: np.ndarray,
    first_origin: int,
    last_origin: int,
    maximum_horizon: int,
    selection_end: int,
    coefficient_fit_end: int,
) -> tuple[np.ndarray, dict]:
    """Forecast every test origin with fixed pre-test ARIMA coefficients.

    Coefficients are selected using observations through ``selection_end`` and
    estimated once using the fixed development window through
    ``coefficient_fit_end``.  Each next observed test value updates the state
    through ``extend`` without refitting parameters.  This preserves rolling
    origins while avoiding a post-hoc or computationally unstable refit at
    every origin.
    """
    from statsmodels.tsa.arima.model import ARIMA

    order, audit, fitted = _select_arima_order(
        values[:selection_end], values[:coefficient_fit_end]
    )
    predictions = []
    state = fitted
    for origin in range(first_origin, last_origin + 1):
        try:
            predictions.append(np.asarray(state.forecast(int(maximum_horizon)), dtype=float))
            state = state.extend(values[origin:origin + 1])
        except Exception as exc:
            raise RuntimeError(f"ARIMA state update failed at origin {origin}: {exc}") from exc
    audit.update({
        "arima_order": list(order),
        "coefficient_fit_end_index": int(coefficient_fit_end),
        "coefficient_fit_converged": _converged(fitted),
        "sequential_update_refit": False,
        "state_update_method": "statsmodels_results_extend",
    })
    return np.asarray(predictions, dtype=float), audit


def write_bundle(root: Path, task: str, model: str, horizon: int, prediction: np.ndarray, truth: np.ndarray, origins: np.ndarray, metadata: dict) -> Path:
    directory = root / "deterministic" / task / model / f"h{horizon}"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz", prediction=prediction, truth=truth, origin_dates=origins)
    atomic_write_json(directory / "meta.json", metadata)
    return directory


def display_path(path: Path) -> str:
    """Store a portable repository-relative path when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def bundle_row(
    root: Path,
    task: str,
    model: str,
    horizon: int,
    entry: dict,
    scale: float,
) -> dict | None:
    """Rebuild a metric row only when an existing deterministic bundle validates."""
    bundle = root / "deterministic" / task / model / f"h{horizon}" / "predictions.npz"
    meta_path = bundle.with_name("meta.json")
    if not bundle.is_file() or not meta_path.is_file():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": task,
        "model_id": model,
        "horizon": int(horizon),
        "evaluation_split": "test",
        "data_sha256": entry["data_sha256"],
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        return None
    payload = np.load(bundle, allow_pickle=False)
    prediction = np.asarray(payload["prediction"], dtype=float)
    truth = np.asarray(payload["truth"], dtype=float)
    origins = payload["origin_dates"].astype(str)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != int(horizon):
        return None
    if len(origins) != prediction.shape[0]:
        return None
    path_values = metrics(truth[:, :, None], prediction[:, :, None], scale)
    terminal_values = metrics(truth[:, -1:, None], prediction[:, -1:, None], scale)
    return {
        "dataset": task,
        "model_id": model,
        "model": {"LastValue": "Last-value", "RandomWalkDrift": "Random-walk-with-drift", "ARIMA": "ARIMA"}[model],
        "source_model_id": model,
        "seed": "deterministic",
        "horizon": int(horizon),
        "origin_count": int(len(origins)),
        "runtime_seconds": float("nan"),
        "peak_gpu_memory_mb": float("nan"),
        **{f"path_{name}": value for name, value in path_values.items()},
        **{f"terminal_{name}": value for name, value in terminal_values.items()},
        "bundle": display_path(bundle),
        "bundle_sha256": sha256_path(bundle),
    }


def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    protocol = json.loads((REPO_ROOT / config["base_protocol"]["path"]).read_text(encoding="utf-8"))
    manifest = json.loads((REPO_ROOT / config["base_protocol"]["training_manifest"]).read_text(encoding="utf-8"))
    tasks = _select(args.tasks, sorted(manifest), "task IDs")
    models = _select(args.models, list(config["deterministic_models"]), "deterministic models")
    available_horizons = [int(value) for value in config["base_protocol"]["horizons"]]
    requested_horizons = [str(value) for value in (args.horizons or available_horizons)]
    horizons = [
        int(value)
        for value in _select(requested_horizons, [str(value) for value in available_horizons], "horizons")
    ]
    if any(not task.endswith("_s") or manifest[task].get("features") != "S" for task in tasks):
        raise ValueError("Deterministic release references require S-only tasks")
    scales = target_scales({task: manifest[task] for task in tasks}, protocol)
    rows_by_key: dict[tuple[str, str, int], dict] = {}
    for task in tasks:
        entry = manifest[task]
        arima_prediction = None
        arima_audit = None
        missing_arima = "ARIMA" in models and any(
            bundle_row(args.output_root, task, "ARIMA", horizon, entry, scales[task]) is None
            for horizon in horizons
        )
        if missing_arima:
            frame = pd.read_csv(REPO_ROOT / entry["root_path"] / entry["data_path"])
            dates = pd.to_datetime(frame["date"], errors="raise")
            selection_end = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["train_end"]), side="right"))
            coefficient_fit_end = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["validation_end"]), side="right"))
            test_start = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_start"]), side="left"))
            test_end = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_end"]), side="right"))
            if coefficient_fit_end != test_start:
                raise ValueError(f"Expected validation fit to end at the first test origin for {task}")
            arima_prediction, arima_audit = arima_forecasts(
                frame[entry["target"]].to_numpy(dtype=float),
                test_start,
                test_end - 1,
                max(horizons),
                selection_end,
                coefficient_fit_end,
            )
        for horizon in horizons:
            values, origins, origin_dates, truth = _series_and_origins(entry, protocol, horizon)
            forecasts: dict[str, tuple[np.ndarray, dict]] = {}
            if "LastValue" in models:
                existing = bundle_row(args.output_root, task, "LastValue", horizon, entry, scales[task])
                if existing is not None:
                    rows_by_key[(task, "LastValue", horizon)] = existing
                else:
                    forecasts["LastValue"] = (last_value(values, origins, horizon), {"method": "last_observed_value"})
            if "RandomWalkDrift" in models:
                existing = bundle_row(args.output_root, task, "RandomWalkDrift", horizon, entry, scales[task])
                if existing is not None:
                    rows_by_key[(task, "RandomWalkDrift", horizon)] = existing
                else:
                    forecasts["RandomWalkDrift"] = (random_walk_drift(values, origins, horizon), {"method": "expanding_history_drift"})
            if "ARIMA" in models:
                existing = bundle_row(args.output_root, task, "ARIMA", horizon, entry, scales[task])
                if existing is not None:
                    rows_by_key[(task, "ARIMA", horizon)] = existing
                else:
                    if arima_prediction is None or arima_audit is None:
                        raise RuntimeError(f"Missing ARIMA forecast cache for {task}/h{horizon}")
                    offset = origins - test_start
                    prediction = arima_prediction[offset, :horizon]
                    forecasts["ARIMA"] = (prediction, {
                        "method": "fixed_parameter_rolling_state_arima",
                        "arima_selection": arima_audit,
                    })
            for model, (prediction, method) in forecasts.items():
                values_path = metrics(truth[:, :, None], prediction[:, :, None], scales[task])
                terminal_values = metrics(truth[:, -1:, None], prediction[:, -1:, None], scales[task])
                metadata = {
                    "protocol_version": config["protocol_version"],
                    "evidence_status": config["evidence_status"],
                    "dataset": task,
                    "data_sha256": entry["data_sha256"],
                    "model_id": model,
                    "horizon": horizon,
                    "evaluation_split": "test",
                    "selection_end": protocol["splits"]["train_end"],
                    "coefficient_fit_end": protocol["splits"]["validation_end"],
                    "statsmodels_version": importlib.metadata.version("statsmodels"),
                    "origin_dates_file": "predictions.npz",
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    **method,
                }
                bundle = write_bundle(args.output_root, task, model, horizon, prediction, truth, origin_dates, metadata)
                rows_by_key[(task, model, horizon)] = {
                    "dataset": task,
                    "model_id": model,
                    "model": {"LastValue": "Last-value", "RandomWalkDrift": "Random-walk-with-drift", "ARIMA": "ARIMA"}[model],
                    "source_model_id": model,
                    "seed": "deterministic",
                    "horizon": horizon,
                    "origin_count": len(origins),
                    "runtime_seconds": float("nan"),
                    "peak_gpu_memory_mb": float("nan"),
                    **{f"path_{name}": value for name, value in values_path.items()},
                    **{f"terminal_{name}": value for name, value in terminal_values.items()},
                    "bundle": display_path(bundle / "predictions.npz"),
                    "bundle_sha256": sha256_path(bundle / "predictions.npz"),
                }
    metrics_root = args.output_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    pd.DataFrame(rows).to_csv(metrics_root / "deterministic_metrics.csv", index=False)
    manifest_payload = {
        "protocol_version": config["protocol_version"],
        "expected_rows": len(tasks) * len(models) * len(horizons),
        "written_rows": len(rows),
        "tasks": tasks,
        "models": models,
        "horizons": horizons,
    }
    atomic_write_json(metrics_root / "deterministic_manifest.json", manifest_payload)
    # The compact metric table is safe to track with the release contract;
    # arrays and checkpoints stay in ignored local output directories.
    release_root = (REPO_ROOT / config["base_protocol"]["path"]).parent / "metrics"
    release_root.mkdir(parents=True, exist_ok=True)
    for name in ("deterministic_metrics.csv", "deterministic_manifest.json"):
        (release_root / name).write_bytes((metrics_root / name).read_bytes())
    expected_rows = len(tasks) * len(models) * len(horizons)
    return {
        "status": "complete" if len(rows) == expected_rows else "partial",
        "rows": len(rows),
        "expected_rows": expected_rows,
        "output": display_path(args.output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks", nargs="+", default=["ALL"])
    parser.add_argument("--models", nargs="+", default=["ALL"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Deterministic reference run blocked: {exc}") from None
