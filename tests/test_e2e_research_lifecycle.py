"""End-to-End integration test covering the entire scientific research lifecycle."""

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hyra.config import HyraConfig
from hyra.engine import get_experiment_status, run_experiment_pipeline
from hyra.formal_verifier import FormalProofReport
from hyra.leakage_detector import analyze_path
from hyra.promoter import check_promotion_readiness, promote_to_project
from hyra.run_tracker import load_run_history
from hyra.scaffolding import init_experiment_scaffold
from hyra.skill_exporter import export_skill


def test_full_scientific_research_lifecycle(tmp_path: Path):
    staging_dir = tmp_path / "staging"
    target_repo = tmp_path / "prod_repo"
    target_repo.mkdir()

    # 1. Initialize Git in target production repo
    subprocess.run(["git", "init"], cwd=target_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Bot"], cwd=target_repo, check=True)
    subprocess.run(["git", "config", "user.email", "bot@test.org"], cwd=target_repo, check=True)
    (target_repo / "README.md").write_text("# Production Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=target_repo, check=True)
    subprocess.run(["git", "commit", "-m", "init: repo setup"], cwd=target_repo, check=True)

    cfg = HyraConfig(
        staging_dir=staging_dir,
        obsidian_vault=None,
        memory_file=None,
        skills_dir=tmp_path / "skills",
        default_project_path=target_repo,
        llm_model="test-model",
    )

    # 2. Step 1: Scaffold experiment workspace
    exp_id = "EXP-999"
    init_info = init_experiment_scaffold(
        exp_id=exp_id,
        title="Adaptive Jump Diffusion SNN",
        project_name="SpikeNets",
        cfg=cfg,
    )
    exp_dir = Path(init_info["staging_dir"])
    assert exp_dir.is_dir()
    assert (exp_dir / "spec.yaml").is_file()

    # 3. Step 2: Simulate Worker developing leaky code and failing AST linter
    leaky_code = (
        "import pandas as pd\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv('test.csv')\n"
        "scaler = StandardScaler()\n"
        "df_scaled = scaler.fit_transform(df)\n"
    )
    model_script = exp_dir / "src" / "model.py"
    model_script.write_text(leaky_code, encoding="utf-8")

    leak_report = analyze_path(exp_dir / "src")
    assert not leak_report.passed
    assert leak_report.critical_count >= 1

    # Promotion must be blocked when code contains data leakage
    (exp_dir / "audit_log.md").write_text("VERDICT: CONFIRMED\n", encoding="utf-8")
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    ready, reason = check_promotion_readiness(exp_dir)
    assert not ready
    assert "Data leakage detected" in reason

    # 4. Step 3: Remediate leaky code to clean pipeline
    clean_code = (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv('test.csv')\n"
        "X = df.drop(columns=['target'])\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled = scaler.transform(X_test)\n"
    )
    model_script.write_text(clean_code, encoding="utf-8")
    leak_report_clean = analyze_path(exp_dir / "src")
    assert leak_report_clean.passed

    # 5. Step 4: Execute reproduction script and verify run tracking
    (exp_dir / "reproduce.sh").write_text("#!/bin/bash\necho 'Simulation successful'\nexit 0\n", encoding="utf-8")
    run_output = asyncio.run(run_experiment_pipeline(exp_id, cfg=cfg))
    assert "Standalone run finished with code 0" in run_output

    runs = load_run_history(exp_dir)
    assert len(runs) == 1
    assert runs[0].experiment_id == exp_id
    assert runs[0].verdict == "CONFIRMED"
    assert runs[0].metrics["cpu_duration_sec"] >= 0.0

    # 6. Step 5: Formal verification gate with Lean 4
    proof_file = exp_dir / "src" / "proof.lean"
    proof_file.write_text("theorem sde_contraction : True := by trivial\n", encoding="utf-8")

    spec = {
        "experiment_id": exp_id,
        "title": "Adaptive Jump Diffusion SNN",
        "project_path": str(target_repo),
        "formal_verification": {
            "enabled": True,
            "engine": "lean4",
            "proof_file": "src/proof.lean",
            "disallow_sorry": True,
        },
        "promotion": {
            "target_module_path": "models/snn_adaptive.py"
        },
    }

    mock_proof_success = FormalProofReport(
        file_path=str(proof_file),
        passed=True,
        has_sorry=False,
    )
    with patch("hyra.promoter.verify_lean_proof", return_value=mock_proof_success):
        ready, reason = check_promotion_readiness(exp_dir, spec=spec)
        assert ready

        # 7. Step 6: Promote confirmed model to target project on Git branch
        (exp_dir / "src" / "snn_adaptive.py").write_text("# Verified model code\n", encoding="utf-8")
        promo_res = promote_to_project(spec, exp_dir, cfg=cfg)
        assert promo_res["git_committed"] is True
        assert f"exp/{exp_id}" in promo_res["git_branch"]
        assert (target_repo / "models" / "snn_adaptive.py").is_file()

    # 8. Step 7: Export skill package for Agensi.io distribution
    dist_dir = tmp_path / "distribution"
    export_rep = export_skill("anti_leakage_researcher", target_platform="agensi", output_dir=dist_dir)
    assert Path(export_rep.archive_path).is_file()
    assert export_rep.size_bytes > 0

    # Check status endpoint
    status = get_experiment_status(exp_id, cfg=cfg)
    assert status["total_runs"] == 1
    assert status["current_status"] == "CONFIRMED"
