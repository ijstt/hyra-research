"""Tests for hyra.stats_gate multi-seed statistical significance gate."""

import pytest

from hyra.stats_gate import (
    SeedComparisonReport,
    compute_bootstrap_ci,
    compute_cohen_d,
    compute_welch_t_test,
    compute_wilcoxon_signed_rank,
    verify_statistical_significance,
)


def test_significant_improvement_paired():
    # 5 matching seeds showing consistent improvement
    candidate = [0.92, 0.95, 0.91, 0.94, 0.93]
    baseline = [0.82, 0.84, 0.80, 0.83, 0.81]

    rep = verify_statistical_significance(
        candidate_scores=candidate,
        baseline_scores=baseline,
        metric_name="accuracy",
        candidate_name="JumpSNN",
        baseline_name="VanillaLIF",
        higher_is_better=True,
    )

    assert rep.is_significant is True
    assert rep.verdict == "SIGNIFICANT_IMPROVEMENT"
    assert rep.mean_delta > 0.10
    assert rep.p_value < 0.05
    assert rep.ci_95_lower > 0.0
    assert rep.effect_size_cohen_d > 2.0

    term = rep.format_terminal()
    assert "HYRA MULTI-SEED STATISTICAL SIGNIFICANCE GATE REPORT" in term
    assert "SIGNIFICANT_IMPROVEMENT" in term
    assert "JumpSNN" in term


def test_no_significant_difference_noise():
    # Minor seed fluctuations with overlapping distributions
    candidate = [0.852, 0.848, 0.851, 0.849, 0.850]
    baseline = [0.850, 0.851, 0.849, 0.852, 0.848]

    rep = verify_statistical_significance(
        candidate_scores=candidate,
        baseline_scores=baseline,
        metric_name="f1",
        higher_is_better=True,
    )

    assert rep.is_significant is False
    assert rep.verdict == "NO_SIGNIFICANT_DIFFERENCE"
    assert rep.p_value >= 0.05
    assert "No statistically significant difference" in rep.summary_text


def test_regression_detected():
    # Candidate degrades performance significantly
    candidate = [0.70, 0.72, 0.69, 0.71, 0.70]
    baseline = [0.88, 0.89, 0.87, 0.90, 0.88]

    rep = verify_statistical_significance(
        candidate_scores=candidate,
        baseline_scores=baseline,
        metric_name="auc_roc",
        higher_is_better=True,
    )

    assert rep.is_significant is False
    assert rep.verdict == "REGRESSION"
    assert "degradation" in rep.summary_text.lower()


def test_lower_is_better_loss_reduction():
    # For loss, lower metric is better
    candidate_loss = [0.12, 0.14, 0.11, 0.13, 0.12]
    baseline_loss = [0.35, 0.38, 0.33, 0.37, 0.36]

    rep = verify_statistical_significance(
        candidate_scores=candidate_loss,
        baseline_scores=baseline_loss,
        metric_name="val_loss",
        higher_is_better=False,
    )

    assert rep.is_significant is True
    assert rep.verdict == "SIGNIFICANT_IMPROVEMENT"
    assert rep.mean_delta < -0.20
    assert rep.ci_95_upper < 0.0


def test_unpaired_welch_t_test():
    # Unequal sample sizes (e.g. 4 vs 5 runs)
    candidate = [0.95, 0.96, 0.94, 0.95]
    baseline = [0.80, 0.82, 0.81, 0.79, 0.81]

    rep = verify_statistical_significance(
        candidate_scores=candidate,
        baseline_scores=baseline,
        is_paired=False,
    )

    assert rep.is_significant is True
    assert "Welch" in rep.test_type
    assert rep.p_value < 0.01


def test_insufficient_samples_error():
    with pytest.raises(ValueError, match="At least 3 seed evaluations"):
        verify_statistical_significance([0.9, 0.8], [0.8, 0.7])
