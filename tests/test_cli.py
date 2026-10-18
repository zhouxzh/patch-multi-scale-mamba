from __future__ import annotations

import json
from pathlib import Path

from pms_mamba.cli import load_config, main


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_identity() -> None:
    payload = load_config(ROOT / "configs" / "pms_mamba_v3.json")
    assert payload["protocol_version"] == "pms_mamba_v3_20260804a"
    assert "PMS_Mamba-PD" in payload["hpo"]["models"]


def test_validate_config_is_read_only(capsys) -> None:
    assert main(["validate-config"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["protocol_version"] == "pms_mamba_v3_20260804a"


def test_preflight_dry_run_is_read_only(capsys, tmp_path: Path) -> None:
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert main(["--data-root", str(tmp_path / "data"), "--results-root", str(tmp_path / "results"), "preflight", "--dry-run"]) == 0
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert before == after
    assert json.loads(capsys.readouterr().out)["writes"] == []
