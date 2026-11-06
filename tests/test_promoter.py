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


def test_promotion_stability_gate_divergent(tmp_path):
    import json

    exp_dir = tmp_path / "exp_stab"
    exp_dir.mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    # Diverging trajectory with NaN/Inf or extreme explosion
    (exp_dir / "results").mkdir()
    traj = [1.0, 10.0, 1000.0, 1e12, float("inf")]
    (exp_dir / "results" / "trajectory.json").write_text(json.dumps(traj), encoding="utf-8")

    spec = {
        "stability_criteria": {
            "enabled": True,
            "trajectory_file": "results/trajectory.json",
            "require_bounded": True,
        }
    }

    ready, reason = check_promotion_readiness(exp_dir, spec=spec)
    assert not ready
    assert "diverges" in reason


def test_promotion_stability_gate_success(tmp_path):
    import json

    exp_dir = tmp_path / "exp_stab_ok"
    exp_dir.mkdir()
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    # Bounded contracting trajectory
    (exp_dir / "results").mkdir()
    traj = [1.0 / (i + 1) for i in range(20)]
    (exp_dir / "results" / "trajectory.json").write_text(json.dumps(traj), encoding="utf-8")

    spec = {
        "stability_criteria": {
            "enabled": True,
            "trajectory_file": "results/trajectory.json",
            "require_bounded": True,
        }
    }

    ready, reason = check_promotion_readiness(exp_dir, spec=spec)
    assert ready
    assert "eligible for promotion" in reason


def test_promote_branch_already_exists(tmp_path):
    target_repo = tmp_path / "target_repo"
    target_repo.mkdir()
    subprocess.run(["git", "init"], cwd=target_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=target_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=target_repo, check=True)
    (target_repo / "README.md").write_text("# Target\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=target_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target_repo, check=True)
    # Pre-create the branch
    subprocess.run(["git", "branch", "exp/EXP-EXISTING"], cwd=target_repo, check=True)

    exp_dir = tmp_path / "EXP-EXISTING"
    exp_dir.mkdir()
    (exp_dir / "src").mkdir()
    (exp_dir / "src" / "algorithm.py").write_text("# code\n", encoding="utf-8")
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    spec = {
        "experiment_id": "EXP-EXISTING",
        "title": "Existing Branch Experiment",
        "project_path": str(target_repo),
        "promotion": {
            "target_module_path": "algorithm.py"
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
    res = promote_to_project(spec, exp_dir, cfg=cfg)
    assert res["git_branch"] == "exp/EXP-EXISTING"
    assert res["git_committed"] is True
    assert (target_repo / "algorithm.py").is_file()


def test_promote_hierarchical_nested_module(tmp_path):
    target_repo = tmp_path / "nested_repo"
    target_repo.mkdir()
    subprocess.run(["git", "init"], cwd=target_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=target_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=target_repo, check=True)
    (target_repo / "README.md").write_text("# Nested\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=target_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=target_repo, check=True)

    exp_dir = tmp_path / "EXP-NESTED"
    exp_dir.mkdir()
    (exp_dir / "src" / "deep" / "models").mkdir(parents=True)
    (exp_dir / "src" / "deep" / "models" / "snn_net.py").write_text("# snn\n", encoding="utf-8")
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    spec = {
        "experiment_id": "EXP-NESTED",
        "title": "Nested Module Promotion",
        "project_path": str(target_repo),
        "promotion": {
            "target_module_path": "deep/models/snn_net.py"
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
    res = promote_to_project(spec, exp_dir, cfg=cfg)
    assert res["git_committed"] is True
    assert (target_repo / "deep" / "models" / "snn_net.py").is_file()

