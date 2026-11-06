import json
import subprocess
import sys
from pathlib import Path


def test_cli_generate_ablations():
    cmd = [
        sys.executable,
        "-m",
        "hyra.cli",
        "generate-ablations",
        "EXP-999",
        "--components",
        "drift",
        "diffusion",
        "jump",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    variants = json.loads(res.stdout)
    assert len(variants) == 7  # 1 full + 3 LOO + 3 isolated
    variant_ids = [v["variant_id"] for v in variants]
    assert "EXP-999_full" in variant_ids
    assert "EXP-999_no_jump" in variant_ids
    assert "EXP-999_only_drift" in variant_ids


def test_cli_analyze_stability(tmp_path):
    traj_file = tmp_path / "traj.json"
    traj_data = [1.0 / (i + 1) for i in range(25)]
    traj_file.write_text(json.dumps(traj_data), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "hyra.cli",
        "analyze-stability",
        str(traj_file),
        "--name",
        "DecayUnit",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(res.stdout)
    assert report["system_name"] == "DecayUnit"
    assert report["is_bounded"] is True
    assert "stability_verdict" in report


def test_cli_stress_search(tmp_path):
    probe_file = tmp_path / "probe.py"
    probe_file.write_text(
        """import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, required=True)
args = parser.parse_args()
stable = args.gamma <= 0.6
metric = 1.0 - args.gamma if stable else 0.05
print(json.dumps({"metric": metric, "is_stable": stable, "failure_mode": None if stable else "Bifurcation"}))
""",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "hyra.cli",
        "stress-search",
        "gamma",
        "--low",
        "0.1",
        "--high",
        "1.0",
        "--baseline",
        "0.9",
        "--script",
        str(probe_file),
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(res.stdout)
    assert report["parameter_name"] == "gamma"
    assert report["critical_threshold"] is not None
    assert 0.4 < report["critical_threshold"] < 0.8


def test_cli_pareto(tmp_path):
    cands_file = tmp_path / "candidates.json"
    cands_data = [
        {"id": "M1", "metrics": {"acc": 0.95, "lat": 15.0}},
        {"id": "M2", "metrics": {"acc": 0.90, "lat": 5.0}},
        {"id": "M3", "metrics": {"acc": 0.88, "lat": 20.0}},  # dominated by M1 & M2
    ]
    cands_file.write_text(json.dumps(cands_data), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "hyra.cli",
        "pareto",
        str(cands_file),
        "--objectives",
        "acc:maximize",
        "lat:minimize",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(res.stdout)
    assert report["total_candidates"] == 3
    assert report["frontier_count"] == 2
    f_ids = {c["candidate_id"] for c in report["frontier_candidates"]}
    assert f_ids == {"M1", "M2"}


def test_cli_stats():
    cmd = [
        sys.executable,
        "-m",
        "hyra.cli",
        "stats",
        "0.92, 0.95, 0.91, 0.94, 0.93",
        "0.82, 0.84, 0.80, 0.83, 0.81",
        "--metric",
        "accuracy",
        "--candidate-name",
        "CandidateSNN",
        "--baseline-name",
        "BaselineSNN",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(res.stdout)
    assert report["is_significant"] is True
    assert report["verdict"] == "SIGNIFICANT_IMPROVEMENT"
    assert report["candidate_name"] == "CandidateSNN"
    assert report["p_value"] < 0.05
