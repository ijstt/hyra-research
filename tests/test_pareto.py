"""Tests for hyra.pareto multi-objective frontier engine."""

import pytest

from hyra.pareto import (
    CandidatePoint,
    Objective,
    ParetoReport,
    compute_pareto_ranking,
    dominates,
    find_pareto_frontier,
)


def test_dominance_relationships():
    acc_obj = Objective("accuracy", "maximize")
    lat_obj = Objective("latency_ms", "minimize")
    objs = [acc_obj, lat_obj]

    # cand_a is strictly superior to cand_b in both accuracy and latency
    cand_a = CandidatePoint("Model_A", {"accuracy": 0.95, "latency_ms": 10.0})
    cand_b = CandidatePoint("Model_B", {"accuracy": 0.90, "latency_ms": 25.0})

    assert dominates(cand_a, cand_b, objs) is True
    assert dominates(cand_b, cand_a, objs) is False

    # cand_c is faster than cand_a, but has lower accuracy: mutually non-dominated trade-off
    cand_c = CandidatePoint("Model_C", {"accuracy": 0.92, "latency_ms": 5.0})
    assert dominates(cand_a, cand_c, objs) is False
    assert dominates(cand_c, cand_a, objs) is False


def test_pareto_ranking_and_frontier():
    objectives = [
        Objective("accuracy", "maximize"),
        Objective("latency_ms", "minimize"),
        Objective("memory_mb", "minimize"),
    ]

    candidates = [
        # Frontier layer 1
        CandidatePoint("Fast_Light", {"accuracy": 0.88, "latency_ms": 5.0, "memory_mb": 100.0}),
        CandidatePoint("High_Accuracy", {"accuracy": 0.97, "latency_ms": 40.0, "memory_mb": 500.0}),
        CandidatePoint("Balanced", {"accuracy": 0.94, "latency_ms": 15.0, "memory_mb": 250.0}),
        # Sub-optimal layer 2 (dominated by Balanced)
        CandidatePoint("Sluggish_Suboptimal", {"accuracy": 0.93, "latency_ms": 30.0, "memory_mb": 350.0}),
        # Completely dominated layer 3 (dominated by Sluggish and Balanced)
        CandidatePoint("Worst_Suboptimal", {"accuracy": 0.85, "latency_ms": 60.0, "memory_mb": 800.0}),
    ]

    report = find_pareto_frontier(candidates, objectives)

    assert report.total_candidates == 5
    assert len(report.frontier_candidates) == 3
    frontier_ids = {c.candidate_id for c in report.frontier_candidates}
    assert frontier_ids == {"Fast_Light", "High_Accuracy", "Balanced"}

    assert len(report.dominated_candidates) == 2
    dominated_ids = {c.candidate_id for c in report.dominated_candidates}
    assert dominated_ids == {"Sluggish_Suboptimal", "Worst_Suboptimal"}

    # Verify layers
    layers = compute_pareto_ranking(candidates, objectives)
    assert len(layers) == 3
    assert {c.candidate_id for c in layers[0]} == {"Fast_Light", "High_Accuracy", "Balanced"}
    assert {c.candidate_id for c in layers[1]} == {"Sluggish_Suboptimal"}
    assert {c.candidate_id for c in layers[2]} == {"Worst_Suboptimal"}

    # Terminal output
    term = report.format_terminal()
    assert "HYRA MULTI-OBJECTIVE PARETO FRONTIER REPORT" in term
    assert "High_Accuracy" in term
    assert "1 (★)" in term
    assert "Worst_Suboptimal" in term


def test_pareto_all_frontier():
    objectives = [
        Objective("gain", "maximize"),
        Objective("cost", "minimize"),
    ]
    candidates = [
        CandidatePoint("P1", {"gain": 1.0, "cost": 10.0}),
        CandidatePoint("P2", {"gain": 2.0, "cost": 20.0}),
        CandidatePoint("P3", {"gain": 3.0, "cost": 30.0}),
    ]
    report = find_pareto_frontier(candidates, objectives)
    assert len(report.frontier_candidates) == 3
    assert len(report.dominated_candidates) == 0
    assert "mutually non-dominated" in report.summary_text


def test_pareto_empty():
    report = find_pareto_frontier([], [Objective("acc", "maximize")])
    assert report.total_candidates == 0
    assert len(report.frontier_candidates) == 0
