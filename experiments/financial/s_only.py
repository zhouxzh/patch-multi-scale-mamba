#!/usr/bin/env python
"""Single entry point for the active S-only financial evidence releases."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from experiments.financial import deterministic, mamba_rerun
from utils.financial import release_setup, s_only_profile, s_only_report
from utils.financial.common import REPO_ROOT


RELEASES = {
    "v1": {
        "config": REPO_ROOT / "configs" / "financial_s_only_v1.json",
        "output": REPO_ROOT / "results" / "financial_s_only_retrospective_v1",
        "checkpoints": REPO_ROOT / "checkpoints" / "financial_s_only_retrospective_v1",
        "release": "financial_s_only_v1",
    },
    "v2": {
        "config": REPO_ROOT / "configs" / "financial_s_only_v2.json",
        "output": REPO_ROOT / "results" / "financial_s_only_retrospective_v2",
        "checkpoints": REPO_ROOT / "checkpoints" / "financial_s_only_retrospective_v2",
        "release": "financial_s_only_v2",
    },
    "v2-supplement": {
        "config": REPO_ROOT / "configs" / "financial_s_only_v2_supplement.json",
        "output": REPO_ROOT / "results" / "financial_s_only_v2_supplement",
        "checkpoints": REPO_ROOT / "checkpoints" / "financial_s_only_v2_supplement",
        "release": "financial_s_only_v2_supplement",
    },
}


def paths(release: str) -> dict:
    return RELEASES[release]


def campaign_args(args: argparse.Namespace) -> Namespace:
    release = paths(args.release)
    return Namespace(
        config=release["config"],
        output_root=release["output"],
        checkpoint_root=release["checkpoints"],
        replay_output_root=(release["output"] / "replay").resolve(),
        tasks=args.tasks,
        models=args.models,
        horizons=args.horizons,
        seeds=args.seeds,
        gpu=args.gpu,
        num_workers=args.num_workers,
        train_epochs=args.train_epochs,
        patience=args.patience,
        max_runs=args.max_runs,
        unlock_retrospective_test=args.unlock_retrospective_test,
    )


def report_args(args: argparse.Namespace) -> Namespace:
    release = paths(args.release)
    return Namespace(
        config=release["config"],
        output_root=release["output"],
        checkpoint_root=release["checkpoints"],
        release=release["release"],
    )


def deterministic_args(args: argparse.Namespace) -> Namespace:
    release = paths(args.release)
    return Namespace(
        config=release["config"],
        output_root=release["output"],
        tasks=args.tasks,
        models=args.deterministic_models,
        horizons=args.horizons,
    )


def profile_args(args: argparse.Namespace) -> Namespace:
    release = paths(args.release)
    return Namespace(
        config=release["config"],
        output_root=release["output"],
        release=release["release"],
        gpu=args.gpu,
        warmup=args.profile_warmup,
        repeats=args.profile_repeats,
        iterations=args.profile_iterations,
    )


def run(args: argparse.Namespace) -> dict:
    if args.stage == "setup":
        return release_setup.setup(args.release)
    if args.stage == "plan":
        config = mamba_rerun.load_config(paths(args.release)["config"])
        task_ids, model_ids, horizons, seeds, manifest = mamba_rerun.resolve_scope(config, campaign_args(args))
        return mamba_rerun.campaign_plan(config, task_ids, model_ids, horizons, seeds, manifest)
    if args.stage == "deterministic":
        return deterministic.run(deterministic_args(args))
    if args.stage == "train":
        return mamba_rerun.train(campaign_args(args))
    if args.stage == "replay":
        return mamba_rerun.replay(campaign_args(args))
    if args.stage == "replay-verify":
        return mamba_rerun.verify_replay(campaign_args(args))
    if args.stage == "replay-summary":
        return mamba_rerun.rebuild_replay_summary(campaign_args(args))
    if args.stage == "profile":
        return s_only_profile.run(profile_args(args))
    if args.stage == "report":
        return s_only_report.build(report_args(args))
    if args.stage == "verify":
        return s_only_report.initialize_release_status(report_args(args))
    if args.stage == "refresh-checksums":
        return s_only_report.refresh_checksums(report_args(args))
    raise ValueError(f"Unsupported S-only stage: {args.stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), default="v1")
    parser.add_argument(
        "--stage",
        choices=(
            "setup", "plan", "deterministic", "train", "replay", "replay-verify",
            "replay-summary", "profile", "report", "verify", "refresh-checksums",
        ),
        required=True,
    )
    parser.add_argument("--tasks", nargs="+", default=["ALL"])
    parser.add_argument("--models", nargs="+", default=["ALL"])
    parser.add_argument("--deterministic-models", nargs="+", default=["ALL"])
    parser.add_argument("--horizons", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2023, 2024, 2025, 2026, 2027])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--unlock-retrospective-test", action="store_true")
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"S-only workflow blocked: {exc}") from None
