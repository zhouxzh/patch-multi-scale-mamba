#!/usr/bin/env python
"""Convert compact release tables to canonical model IDs with source provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.financial.registry import canonical_model_id, public_name


METRICS = REPO_ROOT / "reproducibility" / "financial_reference_evaluation_v1" / "metrics"
MODEL_TABLES = (
    "audited_legacy_baseline_metrics_by_seed.csv",
    "candidate_models_metrics_by_seed.csv",
    "all_model_metrics_by_seed.csv",
    "all_model_aggregate_metrics.csv",
    "balanced_path_rmsse_by_task_seed.csv",
)
CONTRAST_TABLES = (
    "regime_gated_mamba_paired_contrasts.csv",
    "regime_gated_mamba_paired_task_seed_bootstrap.csv",
)


def normalize_model_table(path: Path) -> None:
    frame = pd.read_csv(path)
    if "model_id" not in frame:
        return
    source = frame["source_model_id"] if "source_model_id" in frame else frame["model_id"]
    frame["source_model_id"] = source
    frame["model_id"] = source.map(canonical_model_id)
    frame["model"] = frame["model_id"].map(public_name)
    columns = list(frame.columns)
    columns.remove("source_model_id")
    model_index = columns.index("model_id")
    columns.insert(model_index, "source_model_id")
    frame.to_csv(path, index=False, columns=columns)


def normalize_contrast_table(path: Path) -> None:
    frame = pd.read_csv(path)
    for side in ("candidate", "reference"):
        id_column = f"{side}_id"
        name_column = side
        source_column = f"{side}_source_model_id"
        if id_column not in frame:
            continue
        source = frame[source_column] if source_column in frame else frame[id_column]
        frame[source_column] = source
        frame[id_column] = source.map(canonical_model_id)
        frame[name_column] = frame[id_column].map(public_name)
    frame.to_csv(path, index=False)


def main() -> None:
    for filename in MODEL_TABLES:
        path = METRICS / filename
        if path.is_file():
            normalize_model_table(path)
    for filename in CONTRAST_TABLES:
        path = METRICS / filename
        if path.is_file():
            normalize_contrast_table(path)


if __name__ == "__main__":
    main()
