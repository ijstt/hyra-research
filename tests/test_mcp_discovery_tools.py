import asyncio
import json
import pytest

pytest.importorskip("mcp")
from hyra.mcp_server import create_mcp_server


def test_mcp_server_tools_registered():
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    tool_names = [t.name for t in tools]
    assert "hyra_analyze_stability" in tool_names
    assert "hyra_generate_ablations" in tool_names
    assert "hyra_stress_search" in tool_names


def test_mcp_analyze_stability(tmp_path):
    server = create_mcp_server()
    traj_data = [0.5**i for i in range(15)]
    traj_json = json.dumps(traj_data)

    res = asyncio.run(
        server.call_tool(
            "hyra_analyze_stability",
            {"path_or_json": traj_json, "system_name": "ContractingTest"},
        )
    )
    res_data = json.loads(res.content[0].text)
    assert res_data["system_name"] == "ContractingTest"
    assert res_data["is_bounded"] is True


def test_mcp_generate_ablations():
    server = create_mcp_server()
    res = asyncio.run(
        server.call_tool(
            "hyra_generate_ablations",
            {"exp_id": "EXP-101", "components": "termA, termB, termC"},
        )
    )
    variants = json.loads(res.content[0].text)
    assert len(variants) == 7
    variant_ids = [v["variant_id"] for v in variants]
    assert "EXP-101_full" in variant_ids
    assert "EXP-101_no_termA" in variant_ids


def test_mcp_stress_search(tmp_path):
    server = create_mcp_server()
    probe_file = tmp_path / "probe.py"
    probe_file.write_text(
        """import argparse, json
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, required=True)
args = parser.parse_args()
stable = args.dt <= 0.05
metric = 0.95 if stable else 0.0
print(json.dumps({"metric": metric, "is_stable": stable, "failure_mode": None if stable else "NumericalExplosion"}))
""",
        encoding="utf-8",
    )

    res = asyncio.run(
        server.call_tool(
            "hyra_stress_search",
            {
                "param_name": "dt",
                "low_val": 0.01,
                "high_val": 0.10,
                "baseline_metric": 0.95,
                "script_path": str(probe_file),
            },
        )
    )
    res_data = json.loads(res.content[0].text)
    assert res_data["parameter_name"] == "dt"
    assert res_data["critical_threshold"] is not None
    assert 0.04 <= res_data["critical_threshold"] <= 0.06


def test_mcp_pareto_frontier():
    server = create_mcp_server()
    cands_json = json.dumps([
        {"candidate_id": "NetA", "metrics": {"acc": 0.95, "lat": 20.0}},
        {"candidate_id": "NetB", "metrics": {"acc": 0.90, "lat": 5.0}},
        {"candidate_id": "NetC", "metrics": {"acc": 0.85, "lat": 30.0}},
    ])
    res = asyncio.run(
        server.call_tool(
            "hyra_pareto_frontier",
            {"candidates_json": cands_json, "objectives_str": "acc:maximize lat:minimize"},
        )
    )
    res_data = json.loads(res.content[0].text)
    assert res_data["frontier_count"] == 2
    f_ids = {c["candidate_id"] for c in res_data["frontier_candidates"]}
    assert f_ids == {"NetA", "NetB"}


def test_mcp_verify_significance():
    server = create_mcp_server()
    res = asyncio.run(
        server.call_tool(
            "hyra_verify_significance",
            {
                "candidate_scores": "0.92, 0.95, 0.91, 0.94, 0.93",
                "baseline_scores": "0.82, 0.84, 0.80, 0.83, 0.81",
                "metric_name": "accuracy",
                "higher_is_better": True,
            },
        )
    )
    res_data = json.loads(res.content[0].text)
    assert res_data["is_significant"] is True
    assert res_data["verdict"] == "SIGNIFICANT_IMPROVEMENT"
    assert res_data["p_value"] < 0.05
