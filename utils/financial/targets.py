#!/usr/bin/env python
"""Materialize independent target-only financial series for the v2 release."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.financial.common import REPO_ROOT, RAW_DIR, atomic_write_bytes, atomic_write_json, sha256_path
from utils.financial.prepare import _read_kline_zip, read_fred, read_gld


DEFAULT_PROTOCOL = REPO_ROOT / "reproducibility" / "financial_s_only_v2" / "protocol.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "financial" / "targets"
DEFAULT_MANIFEST = REPO_ROOT / "reproducibility" / "financial_s_only_v2" / "training_datasets.json"


def _clean_series(series: pd.Series, cutoff: str) -> pd.Series:
    result = series.copy()
    result.index = pd.to_datetime(result.index).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    result = result.loc[:pd.Timestamp(cutoff)].replace([np.inf, -np.inf], np.nan).dropna()
    if result.empty:
        raise ValueError(f"Target series {series.name} is empty")
    if not result.index.is_monotonic_increasing or result.index.duplicated().any():
        raise ValueError(f"Target series {series.name} has invalid dates")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"Target series {series.name} contains non-finite values")
    return result.astype(float)


def _binance_start(symbol: str, cutoff: str) -> str:
    root = RAW_DIR / "binance" / symbol
    pattern = re.compile(rf"^{re.escape(symbol)}-1d-(\d{{4}}-\d{{2}})\.zip$")
    months = [match.group(1) for path in root.glob(f"{symbol}-1d-*.zip") if (match := pattern.match(path.name))]
    if not months:
        raise FileNotFoundError(f"No Binance monthly archives found for {symbol} under {root}")
    return min(months)


def _read_binance_target(symbol: str, cutoff: str) -> pd.Series:
    start = _binance_start(symbol, cutoff)
    frames = []
    year, month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in cutoff[:7].split("-"))
    while (year, month) <= (end_year, end_month):
        month_text = f"{year:04d}-{month:02d}"
        path = RAW_DIR / "binance" / symbol / f"{symbol}-1d-{month_text}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Binance archive needed for maximum-history target: {path}")
        frame = _read_kline_zip(path)[["open_time", "close"]].copy()
        timestamps = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
        timestamps_ms = timestamps.where(timestamps < 10**14, timestamps // 1000)
        frame["date"] = pd.to_datetime(timestamps_ms, unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="raise")
        frames.append(frame[["date", "close"]])
        month += 1
        if month == 13:
            year += 1
            month = 1
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    series = frame.set_index("date")["close"].rename(symbol)
    expected = pd.date_range(series.index.min(), pd.Timestamp(cutoff), freq="D")
    missing = expected.difference(series.index)
    if len(missing):
        raise ValueError(f"Unexplained Binance gaps for {symbol}: {missing[:5].strftime('%Y-%m-%d').tolist()}")
    return series


def source_series(protocol: dict) -> dict[str, tuple[pd.Series, str, str]]:
    cutoff = protocol["data_cutoff"]
    output: dict[str, tuple[pd.Series, str, str]] = {}
    # Cross-asset sources.
    output["cross_asset_sp500"] = (read_fred("SP500", cutoff), "FRED:SP500", "level")
    output["cross_asset_wti"] = (read_fred("DCOILWTICO", cutoff), "FRED:DCOILWTICO", "level")
    output["cross_asset_vix"] = (read_fred("VIXCLS", cutoff), "FRED:VIXCLS", "level")
    output["cross_asset_gld"] = (read_gld(cutoff), "Yahoo:GLD", "level")
    # Crypto archives are already UTC daily bars.
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"):
        output[f"crypto_{symbol.lower()}"] = (_read_binance_target(symbol, cutoff), f"Binance:{symbol}", "close")
    # FX source directions match the existing panel preparation.
    fx = {series: read_fred(series, cutoff) for series in ("DEXUSEU", "DEXCAUS", "DEXJPUS", "DEXSZUS", "DEXUSAL", "DEXUSUK")}
    output.update({
        "fx_eurusd": (fx["DEXUSEU"], "FRED:DEXUSEU", "level"),
        "fx_cadusd": (1.0 / fx["DEXCAUS"], "FRED:DEXCAUS", "reciprocal"),
        "fx_jpyusd": (1.0 / fx["DEXJPUS"], "FRED:DEXJPUS", "reciprocal"),
        "fx_chfusd": (1.0 / fx["DEXSZUS"], "FRED:DEXSZUS", "reciprocal"),
        "fx_audusd": (fx["DEXUSAL"], "FRED:DEXUSAL", "level"),
        "fx_gbpusd": (fx["DEXUSUK"], "FRED:DEXUSUK", "level"),
    })
    yields = {series: read_fred(series, cutoff) for series in ("DGS2", "DGS5", "DGS10", "DGS30")}
    for series in yields:
        output[f"yields_{series.lower()}"] = (yields[series], f"FRED:{series}", "level")
    return output


def materialize(protocol_path: Path = DEFAULT_PROTOCOL, output_root: Path = DEFAULT_OUTPUT, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = {"protocol_version", "data_cutoff", "splits", "horizons", "term_groups", "target_roster"}
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Independent-target protocol is missing fields: {missing}")
    sources = source_series(protocol)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for task_id in protocol["target_roster"]:
        if not task_id.startswith("financial_") or not task_id.endswith("_s"):
            raise ValueError(f"Target task must use the active S-only task naming convention: {task_id}")
        source_key = task_id.removeprefix("financial_").removesuffix("_s")
        if source_key not in sources:
            raise ValueError(f"Protocol target has no source mapping: {task_id}")
        series, source_id, transform = sources[source_key]
        series = _clean_series(series.rename(source_key.rsplit("_", 1)[-1].upper()), protocol["data_cutoff"])
        target = str(series.name)
        filename = f"{source_key}.csv"
        path = output_root / filename
        frame = pd.DataFrame({"date": series.index.strftime("%Y-%m-%d"), target: series.to_numpy(dtype=float)})
        atomic_write_bytes(path, frame.to_csv(index=False, float_format="%.10g").encode("utf-8"))
        dates = pd.to_datetime(frame["date"], errors="raise")
        split_counts = {
            "train": int((dates <= pd.Timestamp(protocol["splits"]["train_end"])).sum()),
            "validation": int(
                ((dates >= pd.Timestamp(protocol["splits"]["validation_start"]))
                 & (dates <= pd.Timestamp(protocol["splits"]["validation_end"]))).sum()
            ),
            "test": int(
                ((dates >= pd.Timestamp(protocol["splits"]["test_start"]))
                 & (dates <= pd.Timestamp(protocol["splits"]["test_end"]))).sum()
            ),
        }
        if min(split_counts.values()) <= max(int(value) for value in protocol["horizons"]):
            raise ValueError(f"Insufficient retained observations for {task_id}: {split_counts}")
        calendar_gaps = dates.diff().dt.days.dropna()
        manifest[task_id] = {
            "data": "custom",
            "root_path": str(output_root.relative_to(REPO_ROOT)),
            "data_path": filename,
            "target": target,
            "features": "S",
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "freq": "d",
            "horizons": protocol["horizons"],
            "term_groups": protocol["term_groups"],
            "split_dates": protocol["splits"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": sha256_path(protocol_path),
            "data_sha256": sha256_path(path),
            "source_id": source_id,
            "transform": transform,
            "rows": int(len(frame)),
            "start": str(frame["date"].iloc[0]),
            "end": str(frame["date"].iloc[-1]),
            "split_counts": split_counts,
            "maximum_calendar_gap_days": int(calendar_gaps.max()) if not calendar_gaps.empty else 0,
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_dates": int(frame["date"].duplicated().sum()),
        }
    atomic_write_json(manifest_path, manifest)
    return {"targets": len(manifest), "manifest": str(manifest_path.relative_to(REPO_ROOT)), "output_root": str(output_root.relative_to(REPO_ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(materialize(args.protocol, args.output_root, args.manifest_output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
