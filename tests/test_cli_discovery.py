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
