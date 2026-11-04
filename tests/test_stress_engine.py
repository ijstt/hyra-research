"""Tests for hyra.stress_engine module."""

import pytest

from hyra.stress_engine import (
    PhaseBoundaryReport,
    StressPoint,
    bisection_boundary_search,
    sweep_stress_parameter,
)


def test_bisection_boundary_search_finds_exact_threshold():
    # Simulation model that breaks when noise_sigma >= 1.75
    def mock_simulation(sigma: float):
        if sigma < 1.75:
            # Stable: metric slightly degrades with noise
            metric = 0.90 - 0.05 * sigma
            return metric, True, None
        else:
            # Breakdown: numerical divergence
            return 0.10, False, "DivergenceError"

    report = bisection_boundary_search(
        eval_fn=mock_simulation,
        param_name="noise_sigma",
        low_val=0.0,
        high_val=5.0,
        baseline_metric=0.90,
        tolerance_ratio=0.5,
        max_iterations=15,
        epsilon=1e-3,
    )

    assert report.critical_threshold is not None
    assert 1.74 <= report.critical_threshold <= 1.76
    assert 1.74 <= report.stable_interval[1] <= 1.76
    assert "Phase transition identified" in report.summary_text

    term = report.format_terminal()
    assert "HYRA ADVERSARIAL STRESS & PHASE BOUNDARY REPORT" in term
    assert "DivergenceError" in term


def test_bisection_all_stable():
    def always_stable(param: float):
        return 0.85, True, None

    report = bisection_boundary_search(
        eval_fn=always_stable,
        param_name="perturbation",
        low_val=0.0,
        high_val=1.0,
        baseline_metric=0.85,
    )
    assert report.critical_threshold is None
    assert report.stable_interval == (0.0, 1.0)
    assert "completely stable" in report.summary_text


def test_sweep_stress_parameter():
    # Breaks after dt = 0.2
    def mock_dt_eval(dt: float):
        if dt <= 0.2:
            return 1.0 - dt, True, None
        return 0.0, False, "EulerDivergence"

    report = sweep_stress_parameter(
        eval_fn=mock_dt_eval,
        param_name="dt",
        min_val=0.05,
        max_val=0.5,
        baseline_metric=1.0,
        steps=10,
    )

    assert report.critical_threshold is not None
    assert report.critical_threshold > 0.2
    assert len(report.points_evaluated) == 10
