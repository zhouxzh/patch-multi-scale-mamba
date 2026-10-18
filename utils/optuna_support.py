import json
from pathlib import Path

import numpy as np


DEFAULT_TERM_ORDER = ("short", "middle", "long")
REQUIRED_MANIFEST_FIELDS = (
    "root_path",
    "data_path",
    "target",
    "freq",
    "features",
    "horizons",
    "term_groups",
)
METRIC_FIELDS = ("mae", "mse", "rmse", "mape", "mspe", "rmsse")


def _coerce_int_list(values, field_name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integers") from exc


def _resolve_root(repo_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_dataset_manifest(manifest_path, dataset_slug, repo_root):
    manifest_file = Path(manifest_path)
    with manifest_file.open() as f:
        manifests = json.load(f)

    if dataset_slug not in manifests:
        raise KeyError(
            f"Dataset slug '{dataset_slug}' is not defined in {manifest_file}"
        )

    config = manifests[dataset_slug]
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in config]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_slug}' is missing required manifest fields: {missing}"
        )

    horizons = sorted(set(_coerce_int_list(config["horizons"], "horizons")))
    term_groups = {}
    for term in DEFAULT_TERM_ORDER:
        if term not in config["term_groups"]:
            raise ValueError(f"Dataset '{dataset_slug}' is missing term group '{term}'")
        term_groups[term] = _coerce_int_list(config["term_groups"][term], f"term_groups.{term}")

    manifest_horizon_set = set(horizons)
    grouped_horizon_set = set()
    for term, term_horizons in term_groups.items():
        term_set = set(term_horizons)
        if grouped_horizon_set & term_set:
            overlap = sorted(grouped_horizon_set & term_set)
            raise ValueError(f"Dataset '{dataset_slug}' has overlapping term horizons: {overlap}")
        grouped_horizon_set |= term_set
    if grouped_horizon_set != manifest_horizon_set:
        missing_from_groups = sorted(manifest_horizon_set - grouped_horizon_set)
        extra_in_groups = sorted(grouped_horizon_set - manifest_horizon_set)
        raise ValueError(
            f"Dataset '{dataset_slug}' term groups do not match horizons. "
            f"Missing: {missing_from_groups}, extra: {extra_in_groups}"
        )

    if config["features"] not in ("S", "M", "MS"):
        raise ValueError(
            f"Dataset '{dataset_slug}' has unsupported features='{config['features']}'. Use S, M, or MS."
        )

    root_path = _resolve_root(repo_root, config["root_path"])
    data_file = root_path / config["data_path"]
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_file}")

    return {
        "slug": dataset_slug,
        "data": config.get("data", "custom"),
        "root_path": str(root_path),
        "data_path": config["data_path"],
        "target": config["target"],
        "freq": config["freq"],
        "features": config["features"],
        "horizons": horizons,
        "term_groups": term_groups,
        "enc_in": int(config.get("enc_in", 1)),
        "dec_in": int(config.get("dec_in", 1)),
        "c_out": int(config.get("c_out", 1)),
        "split_dates": config.get("split_dates"),
        "protocol_version": config.get("protocol_version", ""),
        "protocol_sha256": config.get("protocol_sha256", ""),
        "data_sha256": config.get("data_sha256", ""),
        "data_file": str(data_file),
    }


def validate_dataset_file(manifest):
    import pandas as pd

    data_file = Path(manifest["data_file"])
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_file}")

    df = pd.read_csv(data_file, nrows=5)
    if "date" not in df.columns:
        raise ValueError(f"Dataset file is missing required 'date' column: {data_file}")
    if manifest["target"] not in df.columns:
        raise ValueError(
            f"Dataset file is missing target column '{manifest['target']}': {data_file}"
        )


def term_for_horizon(horizon, term_groups):
    for term, horizons in term_groups.items():
        if int(horizon) in horizons:
            return term
    raise KeyError(f"Horizon {horizon} is not assigned to any term group")


def require_term_coverage(horizons, term_groups, label):
    horizon_set = {int(horizon) for horizon in horizons}
    missing_terms = [
        term for term, term_horizons in term_groups.items()
        if not horizon_set.intersection(term_horizons)
    ]
    if missing_terms:
        raise ValueError(
            f"{label} must include at least one horizon from each term group. Missing terms: {missing_terms}"
        )


def term_report_steps(horizons, term_groups):
    term_last_step = {}
    ordered_terms = []
    for step, horizon in enumerate(horizons, start=1):
        term = term_for_horizon(horizon, term_groups)
        term_last_step[term] = step
        if term not in ordered_terms:
            ordered_terms.append(term)
    return [(term_last_step[term], term) for term in ordered_terms]


def summarize_horizon_metrics(horizon_metrics, term_groups):
    if not horizon_metrics:
        raise ValueError("horizon_metrics must not be empty")

    available_horizons = sorted(int(horizon) for horizon in horizon_metrics)
    grouped = {metric: {} for metric in METRIC_FIELDS}

    for metric in METRIC_FIELDS:
        per_horizon_values = [float(horizon_metrics[horizon][metric]) for horizon in available_horizons]
        grouped[metric]["overall"] = float(np.mean(per_horizon_values))
        for term, term_horizons in term_groups.items():
            term_values = [
                float(horizon_metrics[horizon][metric])
                for horizon in available_horizons
                if horizon in term_horizons
            ]
            if term_values:
                grouped[metric][term] = float(np.mean(term_values))

    rmsse_terms = [
        grouped["rmsse"][term]
        for term in DEFAULT_TERM_ORDER
        if term in grouped["rmsse"]
    ]
    if not rmsse_terms:
        raise ValueError("No RMSSE term averages were computed")

    return {
        "available_horizons": available_horizons,
        "grouped_metrics": grouped,
        "balanced_rmsse": float(np.mean(rmsse_terms)),
    }


def json_dumps_stable(value):
    return json.dumps(value, sort_keys=True)
