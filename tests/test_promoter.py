import subprocess
import pytest
from pathlib import Path
from hyra.promoter import check_promotion_readiness, promote_to_project
from hyra.config import HyraConfig


def test_promotion_readiness_missing_audit(tmp_path):
    exp_dir = tmp_path / "exp_01"
    exp_dir.mkdir()
    ready, reason = check_promotion_readiness(exp_dir)
    assert not ready
    assert "audit_log.md" in reason


def test_promotion_readiness_not_confirmed(tmp_path):
    exp_dir = tmp_path / "exp_01"
    exp_dir.mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: FALSIFIED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    ready, reason = check_promotion_readiness(exp_dir)
    assert not ready
    assert "CONFIRMED" in reason


def test_promotion_readiness_missing_reproduce(tmp_path):
    exp_dir = tmp_path / "exp_01"
    exp_dir.mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")

    ready, reason = check_promotion_readiness(exp_dir)
    assert not ready
    assert "reproduce.sh" in reason


def test_promotion_readiness_success(tmp_path):
    exp_dir = tmp_path / "exp_01"
    exp_dir.mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    ready, reason = check_promotion_readiness(exp_dir)
    assert ready


def test_promote_to_project_execution(tmp_path):
    # Setup target repo with git
    target_repo = tmp_path / "mock_repo"
    target_repo.mkdir()
    subprocess.run(["git", "init"], cwd=target_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=target_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=target_repo, check=True)
    (target_repo / "README.md").write_text("# Mock Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=target_repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=target_repo, check=True)

    # Setup experiment staging
    exp_dir = tmp_path / "EXP-777"
    exp_dir.mkdir()
    (exp_dir / "src").mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (exp_dir / "src" / "new_model.py").write_text("# Model code\n", encoding="utf-8")

    spec = {
        "experiment_id": "EXP-777",
        "title": "New Model Experiment",
        "project_path": str(target_repo),
        "promotion": {
            "target_module_path": "models/new_model.py"
        }
    }

    cfg = HyraConfig(
        staging_dir=tmp_path / "staging",
        obsidian_vault=None,
        memory_file=None,
        skills_dir=tmp_path / "skills",
        default_project_path=target_repo,
        llm_model="test",
    )

    result = promote_to_project(spec, exp_dir, cfg=cfg)
    assert result["git_committed"] is True
    assert (target_repo / "models" / "new_model.py").is_file()
    assert (target_repo / "models" / "new_model.py").read_text(encoding="utf-8") == "# Model code\n"
