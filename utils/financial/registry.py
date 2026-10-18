"""Canonical model identities for the final financial-reference release.

Historical experiment identifiers remain in frozen metric snapshots as
``source_model_id`` values. Active commands, configurations, documentation, and
new reports use the mechanism-based canonical identifiers below.
"""

from __future__ import annotations

MODEL_REGISTRY = {
    "RegimeGatedMamba": {
        "public_name": "Regime-Gated Mamba",
        "runtime_model": "RegimeGatedMamba",
        "role": "proposed",
        "source_model_ids": ["PMSV5_R6"],
    },
    "UngatedRegimeLinearMamba": {
        "public_name": "Ungated Regime-Linear Mamba",
        "runtime_model": "RegimeGatedMamba",
        "role": "ablation",
        "source_model_ids": ["PMSV5_R6"],
    },
    "MultiScalePatchMamba": {
        "public_name": "Multi-Scale Patch Mamba",
        "runtime_model": "MultiScalePatchMamba",
        "role": "ablation",
        "source_model_ids": ["PMSV3_B0"],
    },
    "LegacyPatchDecompositionMamba": {
        "public_name": "Legacy Patch-Decomposition Mamba",
        "runtime_model": None,
        "role": "legacy",
        "source_model_ids": ["PMS_Mamba-PD"],
    },
    "InvertedMamba": {
        "public_name": "Inverted Mamba",
        "runtime_model": "InvertedMamba",
        "role": "baseline",
        "source_model_ids": ["S_Mamba"],
    },
    "DecompositionMamba": {
        "public_name": "Decomposition Mamba",
        "runtime_model": "DecompositionMamba",
        "role": "baseline",
        "source_model_ids": ["NS_Mamba-D"],
    },
    "MultiScaleDecompositionMamba": {
        "public_name": "Multi-Scale Decomposition Mamba",
        "runtime_model": "MultiScaleDecompositionMamba",
        "role": "baseline",
        "source_model_ids": ["MS_Mamba-MD"],
    },
    "DLinear": {"public_name": "DLinear", "runtime_model": "DLinear", "role": "baseline", "source_model_ids": ["DLinear"]},
    "NLinear": {"public_name": "NLinear", "runtime_model": "NLinear", "role": "baseline", "source_model_ids": ["NLinear"]},
    "PatchTST": {"public_name": "PatchTST", "runtime_model": "PatchTST", "role": "baseline", "source_model_ids": ["PatchTST"]},
    "TimesNet": {"public_name": "TimesNet", "runtime_model": "TimesNet", "role": "baseline", "source_model_ids": ["TimesNet"]},
    "iTransformer": {"public_name": "iTransformer", "runtime_model": "iTransformer", "role": "baseline", "source_model_ids": ["iTransformer"]},
}

SOURCE_TO_CANONICAL = {}
for _canonical_id, _record in MODEL_REGISTRY.items():
    for _source_id in _record["source_model_ids"]:
        # A source recipe may also underpin an ablation.  Preserve the first
        # (primary) canonical identity when resolving historical records.
        SOURCE_TO_CANONICAL.setdefault(_source_id, _canonical_id)


def canonical_model_id(model_id: str) -> str:
    """Resolve a frozen source identifier to its canonical release identifier."""
    return SOURCE_TO_CANONICAL.get(model_id, model_id)


def source_model_id(model_id: str) -> str:
    """Return the unique frozen source identifier for a canonical model."""
    canonical_id = canonical_model_id(model_id)
    record = MODEL_REGISTRY.get(canonical_id)
    if record is None:
        return model_id
    return record["source_model_ids"][0]


def public_name(model_id: str) -> str:
    """Return the stable display name for canonical or frozen source identifiers."""
    canonical_id = canonical_model_id(model_id)
    return MODEL_REGISTRY.get(canonical_id, {}).get("public_name", canonical_id)


def runtime_model_name(model_id: str) -> str:
    """Return the active ``run.py --model`` name for a canonical model."""
    canonical_id = canonical_model_id(model_id)
    runtime_model = MODEL_REGISTRY.get(canonical_id, {}).get("runtime_model")
    if runtime_model is None:
        raise ValueError(f"{model_id} is historical-only and cannot be trained")
    return runtime_model
