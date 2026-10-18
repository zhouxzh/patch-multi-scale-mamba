#!/usr/bin/env python
"""Recoverably archive inactive mixed S/MS local campaign artifacts.

The active S-only workflow never reads these roots.  This utility records the
pre-move inventory, moves whole campaign roots within the same workspace, and
verifies the inventory after the move.  It intentionally never deletes data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.common import atomic_write_json


ARCHIVE_ROOT = REPO_ROOT / "backup" / "legacy_s_ms_20260812"
ROOTS = (
    Path("results/financial_mamba_rerun_v1"),
    Path("results/financial_baseline_rerun_v1"),
    Path("checkpoints/financial_mamba_rerun_v1"),
    Path("checkpoints/financial_baseline_rerun_v1"),
    Path("results/financial_reference_evaluation_v1"),
    Path("checkpoints/financial_reference_evaluation_v1"),
    Path("results/manuscript_evidence"),
)


def inventory(root: Path) -> dict:
    """Return an inexpensive, deterministic inventory for one complete root."""
    files = []
    byte_count = 0
    newest_mtime_ns = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = path.relative_to(root)
        files.append({
            "path": str(relative),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
        byte_count += stat.st_size
        newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    digest_source = "\n".join(
        f"{row['path']}\t{row['bytes']}\t{row['mtime_ns']}" for row in files
    ).encode("utf-8")
    import hashlib

    return {
        "file_count": len(files),
        "byte_count": byte_count,
        "newest_mtime_ns": newest_mtime_ns,
        "inventory_sha256": hashlib.sha256(digest_source).hexdigest(),
    }


def destination_for(relative: Path) -> Path:
    return ARCHIVE_ROOT / relative


def _entry(relative: Path) -> dict:
    source = REPO_ROOT / relative
    destination = destination_for(relative)
    current = source if source.exists() else destination
    if not current.exists():
        raise FileNotFoundError(f"Neither source nor archive exists: {relative}")
    values = inventory(current)
    return {
        "original_path": str(relative),
        "archive_path": str(destination.relative_to(REPO_ROOT)),
        "restore_to": str(relative),
        "state": "source" if source.exists() else "archived",
        **values,
    }


def archive(execute: bool) -> dict:
    """Optionally move inactive roots and atomically write a verification manifest."""
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    before = [_entry(relative) for relative in ROOTS]
    if execute:
        for relative, entry in zip(ROOTS, before):
            source = REPO_ROOT / relative
            destination = destination_for(relative)
            if not source.exists():
                if not destination.exists():
                    raise FileNotFoundError(source)
                continue
            if destination.exists():
                raise FileExistsError(
                    f"Archive destination already exists for {relative}; inspect the manifest before retrying"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            after = inventory(destination)
            expected = {name: entry[name] for name in ("file_count", "byte_count", "inventory_sha256")}
            observed = {name: after[name] for name in expected}
            if observed != expected:
                raise RuntimeError(f"Archive verification failed for {relative}: {observed} != {expected}")
    entries = [_entry(relative) for relative in ROOTS]
    if any(entry["state"] != "archived" for entry in entries):
        state = "dry_run" if not execute else "partial"
    else:
        state = "archived"
    payload = {
        "archive_version": "legacy_s_ms_20260812",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "move_with_post_move_inventory_verification" if execute else "dry_run",
        "status": state,
        "recovery": "Move each archive_path back to restore_to; verify inventory_sha256 before and after restoration.",
        "entries": entries,
    }
    atomic_write_json(ARCHIVE_ROOT / "archive_manifest.json", payload)
    return {
        "status": state,
        "roots": len(entries),
        "files": sum(entry["file_count"] for entry in entries),
        "bytes": sum(entry["byte_count"] for entry in entries),
        "manifest": str((ARCHIVE_ROOT / "archive_manifest.json").relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Move the inactive roots after creating the inventory.")
    args = parser.parse_args()
    print(json.dumps(archive(args.execute), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
