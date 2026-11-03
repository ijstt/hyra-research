import pytest
from pathlib import Path
from hyra.spec_loader import load_and_validate_spec, SpecValidationError


def test_valid_spec_loading(tmp_path):
    spec_content = """
experiment_id: "EXP-099"
title: "Test Hypothesis"
hypothesis:
  statement: "Test statement"
falsification:
  target_metric: "val_acc"
  baseline_metric: 0.8
  required_gain: 0.05
compute_budget:
  max_runtime_minutes: 10
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(spec_content, encoding="utf-8")

    loaded = load_and_validate_spec(spec_file)
    assert loaded["experiment_id"] == "EXP-099"
    assert loaded["falsification"]["required_gain"] == 0.05


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate_spec(Path("/non/existent/spec.yaml"))


def test_missing_mandatory_key(tmp_path):
    spec_content = """
experiment_id: "EXP-099"
title: "Test Hypothesis"
falsification:
  target_metric: "val_acc"
  baseline_metric: 0.8
  required_gain: 0.05
compute_budget:
  max_runtime_minutes: 10
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(spec_content, encoding="utf-8")

    with pytest.raises(SpecValidationError) as exc:
        load_and_validate_spec(spec_file)
    assert "Missing mandatory specification section: 'hypothesis'" in str(exc.value)


def test_missing_falsification_key(tmp_path):
    spec_content = """
experiment_id: "EXP-099"
title: "Test Hypothesis"
hypothesis:
  statement: "Test"
falsification:
  target_metric: "val_acc"
compute_budget:
  max_runtime_minutes: 10
"""
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(spec_content, encoding="utf-8")

    with pytest.raises(SpecValidationError) as exc:
        load_and_validate_spec(spec_file)
    assert "Missing mandatory field in 'falsification'" in str(exc.value)
