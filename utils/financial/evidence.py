"""Small, release-neutral metric helpers for S-only financial evidence."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils.financial.common import REPO_ROOT


def metrics(truth: np.ndarray, prediction: np.ndarray, scale: float) -> dict[str, float]:
    """Return level-scale MAE/RMSE and train-window scaled RMSSE."""
    error = np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float)
    mse = float(np.mean(np.square(error)))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "rmsse": math.sqrt(mse / float(scale)),
    }


def target_scales(manifest: dict, protocol: dict) -> dict[str, float]:
    """Calculate one RMSSE denominator per target from the training window."""
    output = {}
    train_end = pd.Timestamp(protocol["splits"]["train_end"])
    for task, entry in manifest.items():
        frame = pd.read_csv(REPO_ROOT / entry["root_path"] / entry["data_path"], usecols=["date", entry["target"]])
        train = frame[pd.to_datetime(frame["date"], errors="raise") <= train_end][entry["target"]].to_numpy(dtype=float)
        differences = np.diff(train)
        scale = float(np.mean(np.square(differences)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid RMSSE denominator for {task}")
        output[task] = scale
    return output


def expected_origins(entry: dict, protocol: dict, horizon: int) -> np.ndarray:
    """Return exact rolling test origins for a retained-observation horizon."""
    frame = pd.read_csv(REPO_ROOT / entry["root_path"] / entry["data_path"], usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="raise")
    start = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_start"]), side="left"))
    end = int(dates.searchsorted(pd.Timestamp(protocol["splits"]["test_end"]), side="right"))
    return dates.iloc[np.arange(start, end - int(horizon) + 1)].dt.strftime("%Y-%m-%d").to_numpy()
