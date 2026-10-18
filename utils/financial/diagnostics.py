#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import (
    DEFAULT_PROTOCOL,
    MANIFEST_DIR,
    PROCESSED_DIR,
    atomic_write_bytes,
    atomic_write_json,
    load_protocol,
    sha256_path,
)


QUALITY_REPORT = MANIFEST_DIR / "quality_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Characterize frozen financial nonstationarity.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include final-period diagnostics; use only after final evaluation is unlocked.",
    )
    return parser.parse_args()


def stationarity_tests(values: np.ndarray, adf_regression: str, kpss_regression: str) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 50 or np.std(clean) <= 0:
        raise ValueError("Stationarity diagnostics require at least 50 nonconstant observations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(clean, regression=adf_regression, autolag="AIC")
        kpss_result = kpss(clean, regression=kpss_regression, nlags="auto")
    return {
        "adf_statistic": float(adf[0]),
        "adf_p_value": float(adf[1]),
        "adf_lags": int(adf[2]),
        "kpss_statistic": float(kpss_result[0]),
        "kpss_p_value": float(kpss_result[1]),
        "kpss_lags": int(kpss_result[2]),
    }


def split_summary(values: pd.Series, prefix: str) -> dict:
    differences = values.diff().dropna()
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)),
        f"{prefix}_difference_std": float(differences.std(ddof=1)),
        f"{prefix}_difference_abs_q95": float(differences.abs().quantile(0.95)),
    }


def correlation_shift(panel: pd.DataFrame, target: str, left_mask: pd.Series, right_mask: pd.Series) -> tuple[float, float]:
    features = [column for column in panel.columns if column not in {"date", target}]
    if not features:
        return 0.0, 0.0
    left = panel.loc[left_mask, [*features, target]].diff().corr()[target].drop(target)
    right = panel.loc[right_mask, [*features, target]].diff().corr()[target].drop(target)
    change = (right - left).abs().dropna()
    if change.empty:
        raise ValueError("Unable to estimate cross-series correlation shift")
    return float(change.mean()), float(change.max())


def load_panel_without_locked_targets(name: str, include_test: bool, quality: dict) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if include_test:
        frame = pd.read_csv(path)
    else:
        counts = quality[name]["split_counts"]
        frame = pd.read_csv(path, nrows=int(counts["train"] + counts["validation"]))
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def diagnostic_record(name: str, frame: pd.DataFrame, protocol: dict, include_test: bool) -> dict:
    config = protocol["datasets"][name]
    diagnostics = protocol["nonstationarity_diagnostics"]
    target = config["target"]
    dates = frame["date"]
    train_mask = dates <= pd.Timestamp(protocol["splits"]["train_end"])
    validation_mask = (
        (dates >= pd.Timestamp(protocol["splits"]["validation_start"]))
        & (dates <= pd.Timestamp(protocol["splits"]["validation_end"]))
    )
    train = frame.loc[train_mask, target].astype(float)
    validation = frame.loc[validation_mask, target].astype(float)
    if train.empty or validation.empty:
        raise ValueError(f"Missing train or validation observations for {name}")

    rolling_window = int(diagnostics["rolling_window"])
    trailing_volatility = frame[target].astype(float).diff().rolling(rolling_window).std().shift(1)
    train_rolling = trailing_volatility[train_mask].dropna()
    q_low, q_high = protocol["regime_analysis"]["training_quantiles"]
    result = {
        "dataset": name,
        "target": target,
        "scope": "train_validation_test" if include_test else "train_validation",
        "rolling_window": rolling_window,
        **split_summary(train, "train"),
        **split_summary(validation, "validation"),
        "validation_mean_shift_train_sd": float((validation.mean() - train.mean()) / train.std(ddof=1)),
        "validation_std_ratio": float(validation.std(ddof=1) / train.std(ddof=1)),
        "validation_difference_std_ratio": float(
            validation.diff().dropna().std(ddof=1) / train.diff().dropna().std(ddof=1)
        ),
        "train_trailing_volatility_q25": float(train_rolling.quantile(float(q_low))),
        "train_trailing_volatility_q75": float(train_rolling.quantile(float(q_high))),
        "train_trailing_volatility_min": float(train_rolling.min()),
        "train_trailing_volatility_max": float(train_rolling.max()),
    }
    mean_change, max_change = correlation_shift(frame, target, train_mask, validation_mask)
    result["validation_correlation_shift_mean_abs"] = mean_change
    result["validation_correlation_shift_max_abs"] = max_change
    result.update(
        {f"train_level_{key}": value for key, value in stationarity_tests(
            train.to_numpy(),
            diagnostics["level_tests"]["adf_regression"],
            diagnostics["level_tests"]["kpss_regression"],
        ).items()}
    )
    result.update(
        {f"train_difference_{key}": value for key, value in stationarity_tests(
            train.diff().dropna().to_numpy(),
            diagnostics["difference_tests"]["adf_regression"],
            diagnostics["difference_tests"]["kpss_regression"],
        ).items()}
    )

    if include_test:
        test_mask = (
            (dates >= pd.Timestamp(protocol["splits"]["test_start"]))
            & (dates <= pd.Timestamp(protocol["splits"]["test_end"]))
        )
        test = frame.loc[test_mask, target].astype(float)
        result.update(split_summary(test, "test"))
        result["test_mean_shift_train_sd"] = float((test.mean() - train.mean()) / train.std(ddof=1))
        result["test_std_ratio"] = float(test.std(ddof=1) / train.std(ddof=1))
        result["test_difference_std_ratio"] = float(
            test.diff().dropna().std(ddof=1) / train.diff().dropna().std(ddof=1)
        )
        test_mean_change, test_max_change = correlation_shift(frame, target, train_mask, test_mask)
        result["test_correlation_shift_mean_abs"] = test_mean_change
        result["test_correlation_shift_max_abs"] = test_max_change
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite diagnostic for {name}")
    return result


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    quality_payload = json.loads(QUALITY_REPORT.read_text(encoding="utf-8"))
    quality = {record["dataset"]: record for record in quality_payload["datasets"]}
    records = []
    for name in ("fx", "yields", "cross_asset", "crypto"):
        frame = load_panel_without_locked_targets(name, args.include_test, quality)
        records.append(diagnostic_record(name, frame, protocol, args.include_test))
        print(f"completed nonstationarity diagnostics for {name}")

    suffix = "full" if args.include_test else "selection_safe"
    csv_path = MANIFEST_DIR / f"nonstationarity_diagnostics_{suffix}.csv"
    json_path = MANIFEST_DIR / f"nonstationarity_diagnostics_{suffix}.json"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    atomic_write_bytes(csv_path, buffer.getvalue().encode("utf-8"))
    atomic_write_json(
        json_path,
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256_path(args.protocol),
            "scope": records[0]["scope"],
            "test_targets_read": bool(args.include_test),
            "data_sha256": {
                name: sha256_path(PROCESSED_DIR / f"{name}.csv")
                for name in ("fx", "yields", "cross_asset", "crypto")
            },
            "records": records,
        },
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
