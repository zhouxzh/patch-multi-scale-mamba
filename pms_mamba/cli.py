"""Read-only validation and CPU/model smoke CLI for PMS-Mamba."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config_path(value: str | None) -> Path:
    return (REPO_ROOT / "configs" / "pms_mamba_v3.json") if value is None else Path(value).expanduser().resolve()


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("protocol_version") != "pms_mamba_v3_20260804a":
        raise ValueError("unexpected PMS protocol_version; historical config identity must be preserved")
    required = {"tasks", "horizons", "hpo", "screen", "promotion", "external"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"missing required protocol sections: {', '.join(missing)}")
    if not payload["tasks"] or not payload["horizons"]:
        raise ValueError("protocol must define non-empty tasks and horizons")
    return payload


def _roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "source": REPO_ROOT,
        "data": Path(args.data_root).expanduser().resolve(),
        "results": Path(args.results_root).expanduser().resolve(),
    }


def validate(args: argparse.Namespace) -> int:
    path = _config_path(args.config)
    config = load_config(path)
    print(json.dumps({"config": str(path), "protocol_version": config["protocol_version"],
                      "tasks": len(config["tasks"]), "horizons": config["horizons"]}, indent=2))
    return 0


def preflight(args: argparse.Namespace) -> int:
    path = _config_path(args.config)
    config = load_config(path)
    roots = _roots(args)
    payload = {
        "mode": "dry-run" if args.dry_run else "read-only",
        "repository": str(REPO_ROOT),
        "config": str(path),
        "protocol_version": config["protocol_version"],
        "roots": {key: str(value) for key, value in roots.items()},
        "writes": [],
        "data_root_exists": roots["data"].is_dir(),
        "results_root_exists": roots["results"].is_dir(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def model_smoke(args: argparse.Namespace) -> int:
    try:
        import torch
        from model.PMS_Mamba import Model
    except ImportError as exc:
        print(f"model-smoke unavailable: install the pinned torch and mamba-ssm dependencies ({exc})", file=sys.stderr)
        return 2
    config = SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, output_attention=False,
        use_norm=1, residual_target=0, forecast_head="level",
        increment_loss_weight=0.0, change_gate_regularization=0.0,
        pms_branch_mode="multiscale", ms_use_decomp=0, ms_fusion="mean",
        ms_scales=[1], patch_len=8, stride=4, d_model=16, d_state=8,
        d_ff=32, e_layers=1, dropout=0.0, activation="gelu", enc_in=1, c_out=1,
        moving_avg=5, ns_recent_ratio=0.25, pms_min_patch_tokens=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = Model(config).to(device).eval()
    with torch.no_grad():
        sample = torch.randn(2, args.seq_len, 1, device=device)
        output = model(sample, None, None, None)
    shape = tuple(output[0].shape if isinstance(output, tuple) else output.shape)
    expected = (2, args.pred_len, 1)
    if shape != expected:
        raise RuntimeError(f"unexpected model output shape {shape}, expected {expected}")
    print(json.dumps({"status": "passed", "device": str(device), "output_shape": shape}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pms-mamba", description=__doc__)
    parser.add_argument("--config", help="protocol JSON (default: configs/pms_mamba_v3.json)")
    parser.add_argument("--data-root", default="data", help="external/local data root; never written by read-only commands")
    parser.add_argument("--results-root", default="results", help="ignored results root for future runs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="validate frozen protocol identity and required sections")
    dry = sub.add_parser("preflight", help="inspect roots without writing")
    dry.add_argument("--dry-run", action="store_true", help="assert that this invocation writes nothing")
    smoke = sub.add_parser("model-smoke", help="construct PMS model and run a tiny forward pass")
    smoke.add_argument("--seq-len", type=int, default=32)
    smoke.add_argument("--pred-len", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return validate(args)
    if args.command == "preflight":
        return preflight(args)
    return model_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
