"""Multi-Seed Statistical Significance Gate for Scientific Discovery.

Prevents false-positive discoveries and seed-hunting artifacts by validating
mathematical hypotheses across multiple pseudo-random seeds:
1. Paired Student's t-test for seed-matched experimental evaluations.
2. Welch's two-sample t-test for unequal sample sizes and variances.
3. Wilcoxon signed-rank non-parametric test.
4. Cohen's d effect size estimation.
5. 95% Non-parametric Bootstrap Confidence Intervals for the performance delta.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


@dataclass
class SeedComparisonReport:
    """Comprehensive statistical significance analysis between candidate and baseline across seeds."""
    candidate_name: str
    baseline_name: str
    metric_name: str
    higher_is_better: bool
    n_candidate_seeds: int
    n_baseline_seeds: int
    candidate_mean: float
    candidate_std: float
    baseline_mean: float
    baseline_std: float
    mean_delta: float
    percent_improvement: float
    p_value: float
    test_type: str
    effect_size_cohen_d: float
    ci_95_lower: float
    ci_95_upper: float
    is_significant: bool
    verdict: Literal["SIGNIFICANT_IMPROVEMENT", "NO_SIGNIFICANT_DIFFERENCE", "REGRESSION"]
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_terminal(self) -> str:
        """Render publication-grade ASCII statistical report."""
        lines: List[str] = []
        border = "=" * 80
        lines.append(border)
        lines.append("HYRA MULTI-SEED STATISTICAL SIGNIFICANCE GATE REPORT")
        lines.append(border)

        lines.append(f"Comparison: {self.candidate_name} vs. {self.baseline_name}")
        lines.append(f"Metric: {self.metric_name} | Direction: {'Higher is better' if self.higher_is_better else 'Lower is better'}")
        lines.append(f"Verdict: {self.verdict} | Significant (alpha=0.05): {self.is_significant}")
        lines.append("-" * 80)

        lines.append(f"{'Group':<18} | {'N':<4} | {'Mean':<12} | {'Std Dev':<12}")
        lines.append("-" * 80)
        lines.append(f"{self.candidate_name:<18} | {self.n_candidate_seeds:<4} | {self.candidate_mean:<12.4f} | {self.candidate_std:<12.4f}")
        lines.append(f"{self.baseline_name:<18} | {self.n_baseline_seeds:<4} | {self.baseline_mean:<12.4f} | {self.baseline_std:<12.4f}")
        lines.append("-" * 80)

        delta_sign = "+" if self.mean_delta >= 0 else ""
        lines.append(f"Mean Delta (Candidate - Baseline): {delta_sign}{self.mean_delta:.4f} ({delta_sign}{self.percent_improvement:.2f}%)")
        lines.append(f"95% Bootstrap Confidence Interval: [{self.ci_95_lower:.4f}, {self.ci_95_upper:.4f}]")
        lines.append(f"Hypothesis Test: {self.test_type} | p-value: {self.p_value:.6f}")
        lines.append(f"Standardized Effect Size (Cohen's d): {self.effect_size_cohen_d:.4f}")
        lines.append("-" * 80)
        lines.append(f"Summary: {self.summary_text}")
        lines.append(border)
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: Sequence[float], mean_val: Optional[float] = None) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = mean_val if mean_val is not None else _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def _std_dev(values: Sequence[float], mean_val: Optional[float] = None) -> float:
    return math.sqrt(_variance(values, mean_val))


def compute_cohen_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two independent samples."""
    na, nb = len(sample_a), len(sample_b)
    if na < 2 or nb < 2:
        return 0.0

    ma, mb = _mean(sample_a), _mean(sample_b)
    va, vb = _variance(sample_a, ma), _variance(sample_b, mb)

    pooled_var = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled_var <= 1e-12:
        return 0.0
    return (ma - mb) / math.sqrt(pooled_var)


def compute_bootstrap_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for difference in means (mean_a - mean_b)."""
    na, nb = len(sample_a), len(sample_b)
    if na == 0 or nb == 0:
        return 0.0, 0.0

    rng = random.Random(seed)
    deltas: List[float] = []

    for _ in range(num_resamples):
        resample_a = [sample_a[rng.randint(0, na - 1)] for _ in range(na)]
        resample_b = [sample_b[rng.randint(0, nb - 1)] for _ in range(nb)]
        delta = _mean(resample_a) - _mean(resample_b)
        deltas.append(delta)

    deltas.sort()
    alpha = 1.0 - confidence_level
    idx_lower = int((alpha / 2.0) * num_resamples)
    idx_upper = int((1.0 - alpha / 2.0) * num_resamples)
    idx_upper = min(idx_upper, num_resamples - 1)

    return round(deltas[idx_lower], 4), round(deltas[idx_upper], 4)


def _betacf(a: float, b: float, x: float, max_iter: int = 100, eps: float = 1e-10) -> float:
    """Continued fraction for incomplete beta function (Numerical Recipes)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_p_value(t: float, df: float) -> float:
    """Two-sided p-value for Student's t distribution with df degrees of freedom."""
    if df <= 0.0:
        return 1.0
    x = df / (df + t * t)
    return max(0.0, min(1.0, _incomplete_beta(0.5 * df, 0.5, x)))


def compute_paired_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> Tuple[float, float, float]:
    """Perform Paired Student's t-test on matched evaluations across identical seeds.
    
    Returns:
        Tuple of (t_statistic, degrees_of_freedom, two_sided_p_value).
    """
    n = min(len(sample_a), len(sample_b))
    if n < 2:
        return 0.0, 1.0, 1.0

    diffs = [sample_a[i] - sample_b[i] for i in range(n)]
    mean_d = _mean(diffs)
    var_d = _variance(diffs, mean_d)
    se_d = math.sqrt(var_d / n) if var_d > 0 else 0.0

    if se_d < 1e-12:
        return 0.0, float(n - 1), 1.0 if abs(mean_d) < 1e-12 else 0.0

    t_stat = mean_d / se_d
    df = float(n - 1)
    p_val = _student_t_p_value(t_stat, df)
    return round(t_stat, 4), df, round(p_val, 6)


def compute_welch_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
) -> Tuple[float, float, float]:
    """Perform Welch's two-sample t-test for unequal variances.
    
    Returns:
        Tuple of (t_statistic, degrees_of_freedom, two_sided_p_value).
    """
    na, nb = len(sample_a), len(sample_b)
    if na < 2 or nb < 2:
        return 0.0, 1.0, 1.0

    ma, mb = _mean(sample_a), _mean(sample_b)
    va, vb = _variance(sample_a, ma), _variance(sample_b, mb)

    se_a = va / na
    se_b = vb / nb
    se_diff = math.sqrt(se_a + se_b)

    if se_diff < 1e-12:
        return 0.0, float(na + nb - 2), 1.0 if abs(ma - mb) < 1e-12 else 0.0

    t_stat = (ma - mb) / se_diff

    # Welch-Satterthwaite degrees of freedom
    numerator = (se_a + se_b) ** 2
    denominator = ((se_a ** 2) / (na - 1)) + ((se_b ** 2) / (nb - 1))
    df = numerator / denominator if denominator > 0 else 1.0

    p_value = _student_t_p_value(t_stat, df)
    return round(t_stat, 4), round(df, 2), round(p_value, 6)


def compute_wilcoxon_signed_rank(
    paired_a: Sequence[float],
    paired_b: Sequence[float],
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test on matched paired evaluations across identical seeds.
    
    Returns:
        Tuple of (w_statistic, p_value).
    """
    n = min(len(paired_a), len(paired_b))
    if n < 4:
        return 0.0, 1.0

    diffs = [paired_a[i] - paired_b[i] for i in range(n)]
    non_zero = [(abs(d), d) for d in diffs if abs(d) > 1e-12]
    n_eff = len(non_zero)
    if n_eff < 4:
        return 0.0, 1.0

    non_zero.sort(key=lambda x: x[0])

    ranks: List[float] = [0.0] * n_eff
    i = 0
    while i < n_eff:
        j = i
        while j < n_eff and abs(non_zero[j][0] - non_zero[i][0]) < 1e-12:
            j += 1
        tie_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = tie_rank
        i = j

    w_pos = sum(ranks[k] for k in range(n_eff) if non_zero[k][1] > 0)
    w_neg = sum(ranks[k] for k in range(n_eff) if non_zero[k][1] < 0)
    w_stat = min(w_pos, w_neg)

    mu_w = n_eff * (n_eff + 1) / 4.0
    sigma_w = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0)

    z = (abs(w_pos - mu_w) - 0.5) / sigma_w if sigma_w > 0 else 0.0
    # Two-sided normal approximation
    p_val = _erfc(abs(z) / math.sqrt(2.0))
    p_val = max(0.0, min(1.0, p_val))

    return round(w_stat, 2), round(p_val, 6)


def _erfc(x: float) -> float:
    """Complementary error function approximation."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = (
        0.254829592 * t
        - 0.284496736 * (t**2)
        + 1.421413741 * (t**3)
        - 1.453152027 * (t**4)
        + 1.061405429 * (t**5)
    )
    ans = poly * math.exp(-(x**2))
    return ans if x >= 0 else 2.0 - ans


def verify_statistical_significance(
    candidate_scores: Sequence[float],
    baseline_scores: Sequence[float],
    metric_name: str = "metric",
    candidate_name: str = "Candidate Model",
    baseline_name: str = "Baseline Model",
    higher_is_better: bool = True,
    alpha: float = 0.05,
    is_paired: bool = True,
) -> SeedComparisonReport:
    """Evaluate statistical significance between candidate and baseline runs.
    
    Invariants:
    1. Sample size >= 3 per group required for valid inference.
    2. Significant improvement confirmed ONLY when p < alpha AND 95% CI strictly favors candidate.
    """
    if len(candidate_scores) < 3 or len(baseline_scores) < 3:
        raise ValueError("At least 3 seed evaluations per group are required for statistical significance.")

    c_mean = _mean(candidate_scores)
    c_std = _std_dev(candidate_scores, c_mean)
    b_mean = _mean(baseline_scores)
    b_std = _std_dev(baseline_scores, b_mean)

    mean_delta = c_mean - b_mean
    pct_imp = (mean_delta / abs(b_mean)) * 100 if abs(b_mean) > 1e-12 else 0.0

    cohen_d = compute_cohen_d(candidate_scores, baseline_scores)
    ci_low, ci_high = compute_bootstrap_ci(candidate_scores, baseline_scores)

    # Perform statistical test
    if is_paired and len(candidate_scores) == len(baseline_scores):
        t_stat, df, p_val = compute_paired_t_test(candidate_scores, baseline_scores)
        test_type = f"Paired Student's t-test (t={t_stat}, df={df})"
    else:
        t_stat, df, p_val = compute_welch_t_test(candidate_scores, baseline_scores)
        test_type = f"Welch's Two-Sample t-test (t={t_stat}, df={df})"

    # Determine if improvement is statistically valid
    favorable_delta = (mean_delta > 0) if higher_is_better else (mean_delta < 0)
    unfavorable_delta = (mean_delta < 0) if higher_is_better else (mean_delta > 0)
    ci_excludes_zero = (ci_low > 0) if higher_is_better else (ci_high < 0)
    ci_excludes_zero_regression = (ci_high < 0) if higher_is_better else (ci_low > 0)

    if (p_val < alpha) and favorable_delta and ci_excludes_zero:
        verdict = "SIGNIFICANT_IMPROVEMENT"
        summary = (
            f"{candidate_name} demonstrates a statistically verified improvement over {baseline_name} "
            f"across {len(candidate_scores)} seeds (p = {p_val:.4f} < {alpha}, Cohen's d = {cohen_d:.2f}, "
            f"95% CI: [{ci_low}, {ci_high}])."
        )
    elif (p_val < alpha) and unfavorable_delta and ci_excludes_zero_regression:
        verdict = "REGRESSION"
        summary = (
            f"{candidate_name} exhibits a statistically significant performance degradation "
            f"compared to {baseline_name} (p = {p_val:.4f} < {alpha}, 95% CI: [{ci_low}, {ci_high}])."
        )
    else:
        verdict = "NO_SIGNIFICANT_DIFFERENCE"
        summary = (
            f"No statistically significant difference confirmed between {candidate_name} and {baseline_name} "
            f"(p = {p_val:.4f} >= {alpha}, 95% CI: [{ci_low}, {ci_high}]). Observed variation is consistent with seed noise."
        )

    return SeedComparisonReport(
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        n_candidate_seeds=len(candidate_scores),
        n_baseline_seeds=len(baseline_scores),
        candidate_mean=round(c_mean, 4),
        candidate_std=round(c_std, 4),
        baseline_mean=round(b_mean, 4),
        baseline_std=round(b_std, 4),
        mean_delta=round(mean_delta, 4),
        percent_improvement=round(pct_imp, 2),
        p_value=p_val,
        test_type=test_type,
        effect_size_cohen_d=round(cohen_d, 4),
        ci_95_lower=ci_low,
        ci_95_upper=ci_high,
        is_significant=(verdict == "SIGNIFICANT_IMPROVEMENT"),
        verdict=verdict,
        summary_text=summary,
    )
