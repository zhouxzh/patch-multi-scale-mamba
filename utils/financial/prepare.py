#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils.financial.common import (
    DEFAULT_PROTOCOL,
    MANIFEST_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    REPO_ROOT,
    atomic_write_bytes,
    atomic_write_json,
    load_protocol,
    month_range,
    sha256_path,
)


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen financial benchmark panels.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=MANIFEST_DIR / "training_datasets.json",
        help="Manifest path relative to the repository unless absolute.",
    )
    return parser.parse_args()


def read_fred(series: str, cutoff: str) -> pd.Series:
    path = RAW_DIR / "fred" / f"{series}.csv"
    frame = pd.read_csv(path, na_values=[".", ""])
    if frame.shape[1] != 2:
        raise ValueError(f"Unexpected FRED schema for {series}: {list(frame.columns)}")
    frame.columns = ["date", series]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    frame = frame[frame["date"] <= pd.Timestamp(cutoff)]
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate FRED dates for {series}")
    return frame.set_index("date")[series].sort_index()


def read_gld(cutoff: str) -> pd.Series:
    path = RAW_DIR / "yahoo" / "GLD.csv"
    frame = pd.read_csv(path, usecols=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame[frame["date"] <= pd.Timestamp(cutoff)]
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate GLD dates")
    return frame.set_index("date")["close"].sort_index().rename("GLD")


def _read_kline_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member in {path}, found {members}")
        payload = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(payload), header=None)
    if not frame.empty and str(frame.iloc[0, 0]).lower() in {"open_time", "open time"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.shape[1] != len(KLINE_COLUMNS):
        raise ValueError(f"Unexpected Binance kline schema in {path}: {frame.shape[1]} columns")
    frame.columns = KLINE_COLUMNS
    return frame


def read_binance_symbol(symbol: str, start: str, interval: str, cutoff: str) -> pd.Series:
    frames = []
    for month in month_range(start, cutoff):
        path = RAW_DIR / "binance" / symbol / f"{symbol}-{interval}-{month}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(_read_kline_zip(path)[["open_time", "close"]])
    frame = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    timestamps_ms = timestamps.where(timestamps < 10**14, timestamps // 1000)
    frame["date"] = pd.to_datetime(timestamps_ms, unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame[frame["date"] <= pd.Timestamp(cutoff)]
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate Binance daily bars for {symbol}: {duplicates[:5]}")
    series = frame.set_index("date")["close"].sort_index().rename(symbol)
    expected = pd.date_range(series.index.min(), pd.Timestamp(cutoff), freq="D")
    missing = expected.difference(series.index)
    if len(missing):
        raise ValueError(
            f"Unexplained Binance daily gaps for {symbol}: "
            f"{[value.strftime('%Y-%m-%d') for value in missing[:10]]}"
        )
    return series


def complete_case_panel(
    series: list[pd.Series],
    columns: list[str],
    cutoff: str,
    minimum_start: str | None = None,
) -> pd.DataFrame:
    panel = pd.concat(series, axis=1, join="inner").sort_index()
    panel = panel.loc[:pd.Timestamp(cutoff)]
    if minimum_start:
        panel = panel.loc[pd.Timestamp(minimum_start):]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if panel.index.duplicated().any() or not panel.index.is_monotonic_increasing:
        raise ValueError("Panel dates must be unique and increasing")
    if list(panel.columns) != columns:
        panel = panel[columns]
    return panel


def build_fx(protocol: dict) -> pd.DataFrame:
    config = protocol["datasets"]["fx"]
    source = {series: read_fred(series, protocol["data_cutoff"]) for series in config["source_series"]}
    transformed = {
        "GBPUSD": source["DEXUSUK"],
        "EURUSD": source["DEXUSEU"],
        "CADUSD": 1.0 / source["DEXCAUS"],
        "JPYUSD": 1.0 / source["DEXJPUS"],
        "CHFUSD": 1.0 / source["DEXSZUS"],
        "AUDUSD": source["DEXUSAL"],
    }
    return complete_case_panel(
        [transformed[column].rename(column) for column in config["columns"]],
        config["columns"],
        protocol["data_cutoff"],
        config.get("minimum_start"),
    )


def build_yields(protocol: dict) -> pd.DataFrame:
    config = protocol["datasets"]["yields"]
    source = {series: read_fred(series, protocol["data_cutoff"]) for series in config["source_series"]}
    return complete_case_panel(
        [source[column].rename(column) for column in config["columns"]],
        config["columns"],
        protocol["data_cutoff"],
        config.get("minimum_start"),
    )


def build_cross_asset(protocol: dict) -> pd.DataFrame:
    config = protocol["datasets"]["cross_asset"]
    source = {
        "SP500": read_fred("SP500", protocol["data_cutoff"]).rename("SP500"),
        "WTI": read_fred("DCOILWTICO", protocol["data_cutoff"]).rename("WTI"),
        "VIX": read_fred("VIXCLS", protocol["data_cutoff"]).rename("VIX"),
        "GLD": read_gld(protocol["data_cutoff"]),
    }
    return complete_case_panel(
        [source[column] for column in config["columns"]],
        config["columns"],
        protocol["data_cutoff"],
        config.get("minimum_start"),
    )


def build_crypto(protocol: dict) -> pd.DataFrame:
    config = protocol["datasets"]["crypto"]
    source = {
        symbol: read_binance_symbol(
            symbol,
            config["archive_start"][symbol],
            config["interval"],
            protocol["data_cutoff"],
        )
        for symbol in config["symbols"]
    }
    return complete_case_panel(
        [source[column] for column in config["columns"]],
        config["columns"],
        protocol["data_cutoff"],
        config.get("minimum_start"),
    )


def split_counts(panel: pd.DataFrame, splits: dict) -> dict[str, int]:
    return {
        "train": int((panel.index <= pd.Timestamp(splits["train_end"])).sum()),
        "validation": int(
            ((panel.index >= pd.Timestamp(splits["validation_start"]))
             & (panel.index <= pd.Timestamp(splits["validation_end"]))).sum()
        ),
        "test": int(
            ((panel.index >= pd.Timestamp(splits["test_start"]))
             & (panel.index <= pd.Timestamp(splits["test_end"]))).sum()
        ),
    }


def quality_record(name: str, panel: pd.DataFrame, protocol: dict) -> dict:
    counts = split_counts(panel, protocol["splits"])
    if min(counts.values()) <= max(protocol["horizons"]):
        raise ValueError(f"{name} does not contain enough observations per split: {counts}")
    target = protocol["datasets"][name]["target"]
    target_values = panel[target].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise ValueError(f"Non-finite target values in {name}")
    return {
        "dataset": name,
        "target": target,
        "columns": list(panel.columns),
        "rows": int(len(panel)),
        "start": panel.index.min().date().isoformat(),
        "end": panel.index.max().date().isoformat(),
        "split_counts": counts,
        "missing_cells": int(panel.isna().sum().sum()),
        "duplicate_dates": int(panel.index.duplicated().sum()),
        "maximum_calendar_gap_days": int(panel.index.to_series().diff().dt.days.max()),
        "target_min": float(np.min(target_values)),
        "target_max": float(np.max(target_values)),
        "target_mean": float(np.mean(target_values)),
        "target_std": float(np.std(target_values, ddof=1)),
        "target_first_difference_std": float(np.std(np.diff(target_values), ddof=1)),
    }


def training_entries(
    name: str,
    panel: pd.DataFrame,
    protocol: dict,
    protocol_sha256: str,
) -> dict[str, dict]:
    config = protocol["datasets"][name]
    common = {
        "data": "custom",
        "root_path": "data/financial/processed",
        "data_path": f"{name}.csv",
        "target": config["target"],
        "freq": "d",
        "horizons": protocol["horizons"],
        "term_groups": protocol["term_groups"],
        "split_dates": protocol["splits"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "data_sha256": sha256_path(PROCESSED_DIR / f"{name}.csv"),
    }
    entries = {
        f"financial_{name}_s": {
            **common,
            "features": "S",
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
        },
    }
    return entries


def write_panel(name: str, panel: pd.DataFrame) -> Path:
    output = panel.reset_index(names="date")
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    path = PROCESSED_DIR / f"{name}.csv"
    atomic_write_bytes(path, output.to_csv(index=False, float_format="%.10g").encode("utf-8"))
    return path


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    protocol_sha256 = sha256_path(args.protocol)
    builders = {
        "fx": build_fx,
        "yields": build_yields,
        "cross_asset": build_cross_asset,
        "crypto": build_crypto,
    }
    panels = {name: builder(protocol) for name, builder in builders.items()}
    for name, panel in panels.items():
        write_panel(name, panel)

    quality = [quality_record(name, panel, protocol) for name, panel in panels.items()]
    training_manifest = {}
    for name, panel in panels.items():
        training_manifest.update(
            training_entries(name, panel, protocol, protocol_sha256)
        )

    atomic_write_json(
        MANIFEST_DIR / "quality_report.json",
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol_sha256,
            "data_cutoff": protocol["data_cutoff"],
            "datasets": quality,
        },
    )
    manifest_output = args.manifest_output
    if not manifest_output.is_absolute():
        manifest_output = REPO_ROOT / manifest_output
    atomic_write_json(manifest_output, training_manifest)
    for record in quality:
        print(
            f"{record['dataset']}: rows={record['rows']} "
            f"range={record['start']}..{record['end']} splits={record['split_counts']}"
        )
    print(f"Wrote training manifest: {manifest_output}")


if __name__ == "__main__":
    main()
