from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "configs"
    / "financial_data_sources_v1.json"
)
FINANCIAL_DATA = REPO_ROOT / "data" / "financial"
RAW_DIR = FINANCIAL_DATA / "raw"
PROCESSED_DIR = FINANCIAL_DATA / "processed"
MANIFEST_DIR = FINANCIAL_DATA / "manifests"


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol_version", "data_cutoff", "splits", "horizons", "datasets"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Protocol is missing required fields: {missing}")
    return payload


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded)


def proxy_is_listening(proxy_url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2):
            return True
    except OSError:
        return False


def _request_attempts(
    url: str,
    timeout: tuple[int, int],
    attempts: int,
    proxies: dict[str, str] | None,
) -> requests.Response:
    import requests

    last_error: Exception | None = None
    session = requests.Session()
    session.trust_env = False
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                timeout=timeout,
                proxies=proxies,
                headers={"User-Agent": "nonstationary-mamba-financial-benchmark/1.0"},
            )
            response.raise_for_status()
            return response
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise last_error


def fetch_with_proxy_fallback(
    url: str,
    proxy_config: dict,
    proxy_first: bool = False,
) -> tuple[bytes, dict]:
    import requests

    timeout = (
        int(proxy_config["connect_timeout_seconds"]),
        int(proxy_config["read_timeout_seconds"]),
    )
    proxy_url = str(proxy_config["fallback_url"])
    if proxy_first:
        if not proxy_is_listening(proxy_url):
            raise RuntimeError(f"Proxy-first mode requested but proxy is not listening at {proxy_url}")
        response = _request_attempts(
            url,
            timeout,
            int(proxy_config["proxy_attempts"]),
            proxies={"http": proxy_url, "https": proxy_url},
        )
        transport = "clash-explicit"
    else:
        try:
            response = _request_attempts(
                url,
                timeout,
                int(proxy_config["direct_attempts"]),
                proxies={},
            )
            transport = "direct"
        except (requests.RequestException, OSError) as direct_error:
            if not proxy_is_listening(proxy_url):
                raise RuntimeError(
                    f"Direct download failed and proxy is not listening at {proxy_url}: {direct_error}"
                ) from direct_error
            response = _request_attempts(
                url,
                timeout,
                int(proxy_config["proxy_attempts"]),
                proxies={"http": proxy_url, "https": proxy_url},
            )
            transport = "clash-fallback"

    metadata = {
        "url": url,
        "transport": transport,
        "retrieved_on": date.today().isoformat(),
        "content_length": len(response.content),
        "etag": response.headers.get("ETag", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
        "content_type": response.headers.get("Content-Type", ""),
    }
    return response.content, metadata


def month_range(start: str, end_date: str) -> Iterable[str]:
    start_year, start_month = [int(value) for value in start.split("-")]
    end_year, end_month = [int(value) for value in end_date[:7].split("-")]
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1
