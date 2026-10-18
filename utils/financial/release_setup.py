#!/usr/bin/env python
"""Freeze release-local S-only manifests and validation-selected parameters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from utils.financial.common import REPO_ROOT, atomic_write_json, sha256_path
from utils.financial.registry import MODEL_REGISTRY
from utils.financial.s_only_report import initialize_release_status
from utils.financial.targets import materialize as materialize_targets


SOURCE_MANIFEST = REPO_ROOT / "data" / "financial" / "manifests" / "training_datasets.json"
RELEASES = {
    "v1": REPO_ROOT / "reproducibility" / "financial_s_only_v1",
    "v2": REPO_ROOT / "reproducibility" / "financial_s_only_v2",
    "v2-supplement": REPO_ROOT / "reproducibility" / "financial_s_only_v2_supplement",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mutable_release(release_root: Path) -> None:
    """Do not invalidate the checksum manifest of an already finalized release."""
    if (release_root / "SHA256SUMS.json").is_file():
        raise RuntimeError(
            f"{release_root.name} is finalized and immutable; create a new versioned release "
            "instead of running release setup again"
        )


def _validate_selection_snapshots(release_root: Path, task_names: list[str]) -> list[dict]:
    """Validate the release-local, frozen parameter selections.

    The release contains its own snapshots so setup remains usable after legacy
    mixed-protocol material has been removed from the public repository.
    """
    destination_root = release_root / "selection"
    rows = []
    for task in task_names:
        for filename in ("mamba_best_params.json", "baseline_best_params.json"):
            destination = destination_root / task / filename
            if not destination.is_file():
                raise FileNotFoundError(f"Missing release-local selection snapshot: {destination}")
            rows.append({
                "task": task,
                "file": str(destination.relative_to(release_root)),
                "sha256": sha256_path(destination),
                "bytes": destination.stat().st_size,
                "source": "release-local frozen selection snapshot",
            })
    return rows


def _write_model_map(release_root: Path, protocol: dict) -> None:
    model_ids = [*protocol["learned_models"], *protocol["deterministic_models"]]
    rows = []
    for model_id in model_ids:
        if model_id in MODEL_REGISTRY:
            record = MODEL_REGISTRY[model_id]
            rows.append({
                "model_id": model_id,
                "public_name": record["public_name"],
                "source_model_id": record["source_model_ids"][0],
                "runtime_model": record["runtime_model"],
                "role": record["role"],
            })
        else:
            rows.append({
                "model_id": model_id,
                "public_name": {
                    "LastValue": "Last-value",
                    "RandomWalkDrift": "Random-walk-with-drift",
                    "ARIMA": "ARIMA",
                }[model_id],
                "source_model_id": model_id,
                "runtime_model": None,
                "role": "deterministic_reference",
            })
    atomic_write_json(release_root / "model_identity_map.json", {"models": rows})


def _freeze_v1_manifest(release_root: Path, protocol: dict) -> dict:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            "The local source manifest is missing. Run `python scripts/data.py prepare` "
            "after obtaining the retained raw financial data."
        )
    source = _load(SOURCE_MANIFEST)
    selected = {}
    for task in protocol["tasks"]:
        entry = dict(source.get(task, {}))
        if not entry:
            raise ValueError(f"Source manifest does not contain {task}")
        if entry.get("features") != "S":
            raise ValueError(f"Source manifest task is not target-only: {task}")
        data_path = REPO_ROOT / entry["root_path"] / entry["data_path"]
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing local panel for {task}: {data_path}")
        actual_hash = sha256_path(data_path)
        if actual_hash != entry.get("data_sha256"):
            raise ValueError(f"Source data hash changed for {task}; rebuild the original manifest first")
        frame = pd.read_csv(data_path, usecols=["date", entry["target"]])
        dates = pd.to_datetime(frame["date"], errors="raise")
        calendar_gaps = dates.diff().dt.days.dropna()
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
        entry.update(
            protocol_version=protocol["protocol_version"],
            protocol_sha256=sha256_path(release_root / "protocol.json"),
            selected_input_policy="target_only_S",
            source_panel_manifest=str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
            rows=int(len(frame)),
            start=dates.iloc[0].strftime("%Y-%m-%d"),
            end=dates.iloc[-1].strftime("%Y-%m-%d"),
            split_counts=split_counts,
            maximum_calendar_gap_days=int(calendar_gaps.max()) if not calendar_gaps.empty else 0,
            missing_cells=int(frame.isna().sum().sum()),
            duplicate_dates=int(dates.duplicated().sum()),
        )
        selected[task] = entry
    atomic_write_json(release_root / "training_datasets.json", selected)
    return selected


def _release_readme(protocol: dict, release: str) -> str:
    supplement = protocol.get("evidence_status") == "retrospective_supplement"
    return "\n".join((
        f"# {protocol['protocol_version']}",
        "",
        "This is a local supplementary release contract, not a completed result table." if supplement else "This is a tracked release contract, not a completed result table.",
        "It uses target-only S inputs, frozen pre-2023 family selections, and a",
        "retrospective 2023-01-01 to 2026-06-30 evaluation window.",
        "It supplements V2 gate-ablation evidence and must not be presented as a new primary release." if supplement else "",
        "",
        f"Run `python scripts/data.py release-setup --release {release}` before training.",
        f"Run `python scripts/reproduce.py --workflow financial-s-only --release {release} --stage verify` to inspect completeness.",
        "",
    ))


def setup(release: str) -> dict:
    release_root = RELEASES[release]
    _require_mutable_release(release_root)
    release_root.mkdir(parents=True, exist_ok=True)
    protocol_path = release_root / "protocol.json"
    if release == "v2-supplement" and not protocol_path.is_file():
        shutil.copy2(REPO_ROOT / "configs" / "financial_s_only_v2_supplement_protocol.json", protocol_path)
    protocol = _load(protocol_path)
    if release == "v1":
        manifest = _freeze_v1_manifest(release_root, protocol)
        selection_tasks = list(protocol["tasks"])
    else:
        if release == "v2-supplement":
            source_selection = RELEASES["v2"] / "selection"
            if not source_selection.is_dir():
                raise FileNotFoundError(f"Missing frozen V2 selections: {source_selection}")
            shutil.copytree(source_selection, release_root / "selection", dirs_exist_ok=True)
        if release == "v2-supplement":
            # Reuse the exact V2 input files and hashes; the supplement must not
            # rewrite shared data while creating a local release.
            source_manifest = _load(RELEASES["v2"] / "training_datasets.json")
            manifest = {}
            for task in protocol["target_roster"]:
                if task not in source_manifest:
                    raise ValueError(f"V2 manifest does not contain supplement target: {task}")
                entry = dict(source_manifest[task])
                data_path = REPO_ROOT / entry["root_path"] / entry["data_path"]
                if not data_path.is_file() or sha256_path(data_path) != entry.get("data_sha256"):
                    raise ValueError(f"Frozen V2 input is unavailable or changed for {task}")
                entry.update(
                    protocol_version=protocol["protocol_version"],
                    protocol_sha256=sha256_path(protocol_path),
                    source_release="financial_s_only_v2",
                )
                manifest[task] = entry
            atomic_write_json(release_root / "training_datasets.json", manifest)
        else:
            materialize_targets(
                protocol_path,
                REPO_ROOT / "data" / "financial" / "targets",
                release_root / "training_datasets.json",
            )
            manifest = _load(release_root / "training_datasets.json")
        if set(manifest) != set(protocol["target_roster"]):
            raise ValueError("Independent-target manifest roster differs from the v2 protocol")
        selection_tasks = [
            "financial_cross_asset_s",
            "financial_crypto_s",
            "financial_fx_s",
            "financial_yields_s",
        ]
    selection_rows = _validate_selection_snapshots(release_root, selection_tasks)
    _write_model_map(release_root, protocol)
    config_filename = "financial_s_only_v2_supplement.json" if release == "v2-supplement" else f"financial_s_only_{release}.json"
    config_source = REPO_ROOT / "configs" / config_filename
    config_destination = release_root / "config.json"
    shutil.copy2(config_source, config_destination)
    if sha256_path(config_source) != sha256_path(config_destination):
        raise RuntimeError(f"Configuration snapshot checksum mismatch: {config_destination}")
    (release_root / "README.md").write_text(_release_readme(protocol, release), encoding="utf-8")
    atomic_write_json(release_root / "selection_manifest.json", {
        "protocol_version": protocol["protocol_version"],
        "selection_policy": protocol["selection_policy"],
        "entries": selection_rows,
    })
    # The release is intentionally incomplete until every learned run is retained.
    # Record this explicitly instead of inheriting an old campaign's status.
    from argparse import Namespace

    release_key = "financial_s_only_v2_supplement" if release == "v2-supplement" else f"financial_s_only_{release}"
    output_name = "financial_s_only_v2_supplement" if release == "v2-supplement" else f"financial_s_only_retrospective_{release}"
    initialize_release_status(Namespace(
        config=config_source,
        output_root=REPO_ROOT / "results" / output_name,
        checkpoint_root=REPO_ROOT / "checkpoints" / output_name,
        release=release_key,
    ))
    return {
        "release": release,
        "protocol_version": protocol["protocol_version"],
        "tasks": len(manifest),
        "selection_files": len(selection_rows),
        "config_sha256": sha256_path(config_destination),
        "manifest": str((release_root / "training_datasets.json").relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    args = parser.parse_args()
    print(json.dumps(setup(args.release), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
