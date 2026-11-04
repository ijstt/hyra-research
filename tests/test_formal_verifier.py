"""Tests for Lean 4 formal verification engine and promotion gate integration."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hyra.formal_verifier import (
    FormalProofReport,
    LeanDiagnostic,
    _parse_lean_output,
    is_lean_installed,
    verify_lean_proof,
)
from hyra.promoter import check_promotion_readiness
from hyra.spec_loader import SpecValidationError, load_and_validate_spec


def test_parse_lean_output_json():
    stdout = (
        '{"file_name": "test.lean", "pos": {"line": 10, "column": 2}, "severity": "error", "data": "type mismatch"}\n'
        '{"file_name": "test.lean", "pos": {"line": 15, "column": 4}, "severity": "warning", "data": "declaration uses \'sorry\'"}\n'
    )
    diagnostics = _parse_lean_output(stdout, "", "test.lean")
    assert len(diagnostics) == 2
    assert diagnostics[0].severity == "ERROR"
    assert diagnostics[0].line == 10
    assert diagnostics[0].col == 2
    assert "type mismatch" in diagnostics[0].message
    assert diagnostics[1].severity == "WARNING"
    assert diagnostics[1].line == 15
    assert "sorry" in diagnostics[1].message


def test_parse_lean_output_text():
    text = (
        "src/proof.lean:4:1: error: unknown identifier 'x'\n"
        "src/proof.lean:8:12: warning: declaration uses 'sorry'\n"
    )
    diagnostics = _parse_lean_output(text, "", "src/proof.lean")
    assert len(diagnostics) == 2
    assert diagnostics[0].line == 4
    assert diagnostics[0].col == 1
    assert diagnostics[0].severity == "ERROR"
    assert diagnostics[1].line == 8
    assert diagnostics[1].severity == "WARNING"


def test_verify_missing_proof_file(tmp_path: Path):
    missing_file = tmp_path / "non_existent.lean"
    report = verify_lean_proof(missing_file)
    assert not report.passed
    assert "not found" in (report.error_message or "").lower()


def test_verify_when_lean_not_installed(tmp_path: Path):
    proof_file = tmp_path / "simple.lean"
    proof_file.write_text("theorem trivial_id (p : Prop) : p -> p := id\n", encoding="utf-8")

    with patch("hyra.formal_verifier.is_lean_installed", return_value=False):
        report = verify_lean_proof(proof_file)
        assert not report.passed
        assert not report.lean_installed
        term = report.format_terminal()
        assert "BLOCKED" in term
        assert "elan" in term


def test_verify_lean_success_mock(tmp_path: Path):
    proof_file = tmp_path / "proof.lean"
    proof_file.write_text("theorem add_comm (n m : Nat) : n + m = m + n := Nat.add_comm n m\n", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""

    with patch("hyra.formal_verifier.is_lean_installed", return_value=True):
        with patch("subprocess.run", return_value=mock_proc):
            report = verify_lean_proof(proof_file)
            assert report.passed
            assert not report.has_sorry
            assert len(report.diagnostics) == 0
            assert "PASSED" in report.format_terminal()


def test_verify_lean_sorry_detected(tmp_path: Path):
    proof_file = tmp_path / "incomplete.lean"
    proof_file.write_text("theorem unproved_thm : 1 + 1 = 3 := by sorry\n", encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = '{"file_name": "incomplete.lean", "pos": {"line": 1, "column": 0}, "severity": "warning", "data": "declaration uses \'sorry\'"}\n'
    mock_proc.stderr = ""

    with patch("hyra.formal_verifier.is_lean_installed", return_value=True):
        with patch("subprocess.run", return_value=mock_proc):
            # Strict mode: disallow sorry (default)
            report_strict = verify_lean_proof(proof_file, disallow_sorry=True)
            assert not report_strict.passed
            assert report_strict.has_sorry

            # Permissive mode: allow sorry
            report_perm = verify_lean_proof(proof_file, disallow_sorry=False)
            assert report_perm.passed
            assert report_perm.has_sorry


def test_verify_lean_timeout(tmp_path: Path):
    proof_file = tmp_path / "slow.lean"
    proof_file.write_text("theorem slow_thm : True := by trivial\n", encoding="utf-8")

    with patch("hyra.formal_verifier.is_lean_installed", return_value=True):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="lean", timeout=5)):
            report = verify_lean_proof(proof_file, timeout_sec=5)
            assert not report.passed
            assert "timed out" in (report.error_message or "").lower()


def test_verify_unauthorized_axiom(tmp_path: Path):
    proof_file = tmp_path / "axiom_hack.lean"
    proof_file.write_text("axiom fake_truth : False\ntheorem my_thm : False := fake_truth\n", encoding="utf-8")

    mock_proc_compile = MagicMock()
    mock_proc_compile.returncode = 0
    mock_proc_compile.stdout = ""
    mock_proc_compile.stderr = ""

    mock_proc_axiom = MagicMock()
    mock_proc_axiom.returncode = 0
    mock_proc_axiom.stdout = "'my_thm' depends on axioms: [fake_truth, propext]"
    mock_proc_axiom.stderr = ""

    with patch("hyra.formal_verifier.is_lean_installed", return_value=True):
        with patch("subprocess.run", side_effect=[mock_proc_compile, mock_proc_axiom]):
            report = verify_lean_proof(proof_file, target_theorems=["my_thm"])
            assert not report.passed
            assert "fake_truth" in report.unauthorized_axioms


def test_spec_loader_formal_verification_validation(tmp_path: Path):
    spec_file = tmp_path / "spec.yaml"
    spec_content = """
experiment_id: "EXP-010"
title: "Formal SDE Stability"
hypothesis:
  statement: "Investigate stability"
falsification:
  target_metric: "val_loss"
  baseline_metric: 1.0
  required_gain: 0.05
compute_budget:
  max_runtime_minutes: 30
formal_verification:
  enabled: true
  engine: "lean4"
  proof_file: "src/stability.lean"
"""
    spec_file.write_text(spec_content, encoding="utf-8")
    data = load_and_validate_spec(spec_file)
    assert data["formal_verification"]["enabled"] is True

    # Test error when enabled without proof_file
    invalid_spec = spec_content.replace('proof_file: "src/stability.lean"', "")
    spec_file.write_text(invalid_spec, encoding="utf-8")
    with pytest.raises(SpecValidationError, match="Missing 'proof_file'"):
        load_and_validate_spec(spec_file)


def test_promotion_gate_with_formal_verification(tmp_path: Path):
    exp_dir = tmp_path / "EXP-FORMAL"
    exp_dir.mkdir()
    (exp_dir / "src").mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    spec = {
        "experiment_id": "EXP-FORMAL",
        "formal_verification": {
            "enabled": True,
            "engine": "lean4",
            "proof_file": "src/proof.lean",
            "disallow_sorry": True,
        },
    }

    # Case 1: Missing proof file
    ready, reason = check_promotion_readiness(exp_dir, spec=spec)
    assert not ready
    assert "proof file 'src/proof.lean' was not found" in reason

    # Case 2: Proof file with sorry
    proof_file = exp_dir / "src" / "proof.lean"
    proof_file.write_text("theorem unproved : True := by sorry\n", encoding="utf-8")

    mock_fail_report = FormalProofReport(
        file_path=str(proof_file),
        passed=False,
        has_sorry=True,
        error_message="Contains sorry",
    )
    with patch("hyra.promoter.verify_lean_proof", return_value=mock_fail_report):
        ready, reason = check_promotion_readiness(exp_dir, spec=spec)
        assert not ready
        assert "Formal verification gate failed" in reason

    # Case 3: Proof valid
    mock_pass_report = FormalProofReport(
        file_path=str(proof_file),
        passed=True,
        has_sorry=False,
    )
    with patch("hyra.promoter.verify_lean_proof", return_value=mock_pass_report):
        ready, reason = check_promotion_readiness(exp_dir, spec=spec)
        assert ready


def test_cli_verify_proof_command(tmp_path: Path):
    proof_file = tmp_path / "test_cmd.lean"
    proof_file.write_text("theorem id_prop (p : Prop) : p -> p := id\n", encoding="utf-8")

    # CLI call without lean installed returns code 1
    res = subprocess.run(
        [sys.executable, "-m", "hyra.cli", "verify-proof", str(proof_file), "--json"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    data = json.loads(res.stdout)
    assert data["passed"] is False
    assert data["lean_installed"] is False
