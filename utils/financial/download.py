#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from utils.financial.common import (
    DEFAULT_PROTOCOL,
    MANIFEST_DIR,
    RAW_DIR,
    REPO_ROOT,
    atomic_write_bytes,
    atomic_write_json,
    fetch_with_proxy_fallback,
    load_protocol,
    month_range,
    proxy_is_listening,
    sha256_path,
)


FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=1900-01-01&coed={cutoff}"
BINANCE_MONTHLY = (
    "https://data.binance.vision/data/spot/monthly/klines/"
    "{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download frozen financial benchmark sources.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--refresh", action="store_true", help="Redownload existing FRED and Yahoo files.")
    parser.add_argument(
        "--proxy-first",
        action="store_true",
        help="Use the configured Clash proxy immediately instead of attempting a direct connection.",
    )
    parser.add_argument(
        "--skip-binance",
        action="store_true",
        help="Skip Binance archives while testing the FRED/Yahoo path.",
    )
    parser.add_argument(
        "--skip-yahoo",
        action="store_true",
        help="Skip Yahoo GLD; useful when the current interpreter does not provide yfinance.",
    )
    parser.add_argument(
        "--only-yahoo",
        action="store_true",
        help="Download only Yahoo GLD and preserve any existing manifest records.",
    )
    return parser.parse_args()


def file_record(path: Path, metadata: dict, source_id: str, rights: str) -> dict:
    return {
        "source_id": source_id,
        "path": str(path.relative_to(RAW_DIR.parent)),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "rights_note": rights,
        **metadata,
    }


def download_fred(protocol: dict, refresh: bool, proxy_first: bool) -> list[dict]:
    series_ids = set(protocol["datasets"]["fx"]["source_series"])
    series_ids.update(protocol["datasets"]["yields"]["source_series"])
    series_ids.update(protocol["datasets"]["cross_asset"]["fred_series"])
    rights_by_series = {
        "SP500": "Copyrighted; pre-approval required for reproduction. Keep raw values local.",
        "VIXCLS": "Copyrighted; citation required. Keep raw values local.",
        "DCOILWTICO": "Public domain; citation requested.",
    }
    records = []
    for series in sorted(series_ids):
        path = RAW_DIR / "fred" / f"{series}.csv"
        url = FRED_CSV.format(series=series, cutoff=protocol["data_cutoff"])
        metadata = {"url": url, "transport": "existing", "retrieved_on": ""}
        if refresh or not path.exists():
            payload, metadata = fetch_with_proxy_fallback(url, protocol["proxy"], proxy_first)
            if len(payload) < 20 or b"DATE" not in payload[:200].upper():
                raise ValueError(f"FRED returned an invalid CSV for {series}")
            atomic_write_bytes(path, payload)
        rights = rights_by_series.get(series, "Public domain; citation requested.")
        records.append(file_record(path, metadata, series, rights))
        print(f"FRED {series}: {path} ({records[-1]['sha256'][:12]})")
    return records


def _normalize_yahoo_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no rows for {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol in set(frame.columns.get_level_values(-1)):
            frame = frame.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            frame.columns = [str(column[0]) for column in frame.columns]
    frame = frame.reset_index()
    date_column = "Date" if "Date" in frame.columns else frame.columns[0]
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Yahoo GLD response is missing columns: {missing}")
    normalized = frame.rename(columns={date_column: "date", "Close": "close"})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime("%Y-%m-%d")
    normalized = normalized[["date", "close", "High", "Low", "Open", "Volume"]]
    normalized = normalized.dropna(subset=["date", "close", "High", "Low", "Open"])
    normalized = normalized.sort_values("date").drop_duplicates("date", keep="last")
    return normalized.reset_index(drop=True)


def _yahoo_download(symbol: str, start: str, end_exclusive: str, proxy_url: str | None) -> pd.DataFrame:
    import yfinance as yf

    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    original = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            if proxy_url:
                os.environ[key] = proxy_url
            else:
                os.environ.pop(key, None)
        frame = yf.download(
            symbol,
            start=start,
            end=end_exclusive,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=90,
        )
        return _normalize_yahoo_frame(frame, symbol)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def download_gld(protocol: dict, refresh: bool, proxy_first: bool) -> list[dict]:
    config = protocol["datasets"]["cross_asset"]
    symbol = config["yahoo_symbol"]
    path = RAW_DIR / "yahoo" / f"{symbol}.csv"
    metadata = {
        "url": f"https://finance.yahoo.com/quote/{symbol}/history/",
        "transport": "existing",
        "retrieved_on": "",
    }
    if refresh or not path.exists():
        cutoff = datetime.strptime(protocol["data_cutoff"], "%Y-%m-%d").date()
        end_exclusive = (cutoff + timedelta(days=1)).isoformat()
        try:
            if proxy_first:
                raise RuntimeError("proxy-first requested")
            frame = _yahoo_download(symbol, "2004-11-18", end_exclusive, None)
            transport = "direct"
        except Exception as direct_error:
            proxy_url = protocol["proxy"]["fallback_url"]
            if not proxy_is_listening(proxy_url):
                raise RuntimeError(
                    f"Yahoo direct download failed and Clash is unavailable: {direct_error}"
                ) from direct_error
            frame = _yahoo_download(symbol, "2004-11-18", end_exclusive, proxy_url)
            transport = "clash-fallback"
        payload = frame.to_csv(index=False, float_format="%.8f").encode("utf-8")
        atomic_write_bytes(path, payload)
        metadata.update(transport=transport, retrieved_on=datetime.now().date().isoformat())
    return [
        file_record(
            path,
            metadata,
            f"Yahoo:{symbol}",
            "Yahoo Finance source; keep raw values local pending terms review.",
        )
    ]


def _expected_checksum(payload: bytes, filename: str) -> str:
    text = payload.decode("utf-8").strip()
    fields = text.split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"Invalid Binance checksum file for {filename}")
    if len(fields) > 1 and Path(fields[-1].lstrip("*" )).name != filename:
        raise ValueError(f"Binance checksum filename mismatch for {filename}: {fields[-1]}")
    return fields[0].lower()


def download_binance(protocol: dict, proxy_first: bool) -> list[dict]:
    config = protocol["datasets"]["crypto"]
    records = []
    for symbol in config["symbols"]:
        for month in month_range(config["archive_start"][symbol], protocol["data_cutoff"]):
            filename = f"{symbol}-{config['interval']}-{month}.zip"
            path = RAW_DIR / "binance" / symbol / filename
            checksum_path = path.with_suffix(path.suffix + ".CHECKSUM")
            url = BINANCE_MONTHLY.format(
                symbol=symbol,
                interval=config["interval"],
                month=month,
            )
            metadata = {"url": url, "transport": "existing", "retrieved_on": ""}
            if not checksum_path.exists():
                checksum_payload, checksum_meta = fetch_with_proxy_fallback(
                    url + ".CHECKSUM", protocol["proxy"], proxy_first
                )
                atomic_write_bytes(checksum_path, checksum_payload)
            else:
                checksum_payload = checksum_path.read_bytes()
                checksum_meta = {"url": url + ".CHECKSUM", "transport": "existing", "retrieved_on": ""}
            expected = _expected_checksum(checksum_payload, filename)
            if not path.exists() or sha256_path(path) != expected:
                payload, metadata = fetch_with_proxy_fallback(url, protocol["proxy"], proxy_first)
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Binance checksum mismatch after download: {filename}")
                atomic_write_bytes(path, payload)
            if sha256_path(path) != expected:
                raise ValueError(f"Binance checksum mismatch: {filename}")
            records.append(
                file_record(
                    path,
                    metadata,
                    f"Binance:{symbol}:{month}",
                    "Official Binance public archive; verify data terms before redistribution.",
                )
            )
            records.append(
                file_record(
                    checksum_path,
                    checksum_meta,
                    f"Binance-checksum:{symbol}:{month}",
                    "Checksum metadata distributed with the Binance archive.",
                )
            )
            print(f"Binance {symbol} {month}: verified")
    return records


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    manifest_path = MANIFEST_DIR / "download_manifest.json"
    records = []
    if args.only_yahoo and manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8")).get("records", [])
        records = [record for record in records if record.get("source_id") != "Yahoo:GLD"]
    if not args.only_yahoo:
        records.extend(download_fred(protocol, args.refresh, args.proxy_first))
    if not args.skip_yahoo:
        records.extend(download_gld(protocol, args.refresh, args.proxy_first))
    if not args.skip_binance and not args.only_yahoo:
        records.extend(download_binance(protocol, args.proxy_first))
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "data_cutoff": protocol["data_cutoff"],
        "path_base": str(RAW_DIR.parent.relative_to(REPO_ROOT)),
        "record_count": len(records),
        "records": records,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Wrote {len(records)} source records to {manifest_path}")


if __name__ == "__main__":
    main()
