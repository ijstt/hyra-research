"""Adversarial Boundary and Stress Search Engine for physical parameter limits.

Automatically probes the boundaries of mathematical formulations and AI models:
1. Locates the critical threshold theta_c (noise intensity sigma_c, time step dt_max, perturbation epsilon)
   where a system undergoes phase transition or catastrophic failure.
2. Performs bisection search to discover exact phase boundaries with arbitrary precision.
3. Generates phase stability reports and textual phase boundary diagrams.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class StressPoint:
    parameter_value: float
    metric_value: float
    is_stable: bool
    failure_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseBoundaryReport:
    parameter_name: str
    baseline_param_value: float
    baseline_metric: float
    critical_threshold: Optional[float]
    tolerance_ratio: float
    points_evaluated: List[StressPoint] = field(default_factory=list)
    stable_interval: Tuple[float, float] = (0.0, 0.0)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "baseline_param_value": self.baseline_param_value,
            "baseline_metric": self.baseline_metric,
            "critical_threshold": self.critical_threshold,
            "tolerance_ratio": self.tolerance_ratio,
            "stable_interval": list(self.stable_interval),
            "points_count": len(self.points_evaluated),
            "points_evaluated": [p.to_dict() for p in self.points_evaluated],
            "summary_text": self.summary_text,
        }

    def format_terminal(self) -> str:
        lines: List[str] = []
        border = "=" * 78
        lines.append(border)
        lines.append(f"HYRA ADVERSARIAL STRESS & PHASE BOUNDARY REPORT: {self.parameter_name}")
        lines.append(border)
        lines.append(f"Baseline Parameter: {self.baseline_param_value:.4f} | Baseline Metric: {self.baseline_metric:.4f}")

        thresh_str = f"{self.critical_threshold:.4f}" if self.critical_threshold is not None else "NOT FOUND IN RANGE"
        lines.append(f"Critical Breakdown Threshold: {thresh_str}")
        lines.append(f"Verified Stable Interval: [{self.stable_interval[0]:.4f}, {self.stable_interval[1]:.4f}]")
        lines.append("-" * 78)
        lines.append(f"{self.parameter_name:<18} | {'Metric':<10} | {'Status':<10} | {'Failure Mode'}")
        lines.append("-" * 78)

        for p in self.points_evaluated:
            status = "STABLE" if p.is_stable else "FAILED"
            mode = p.failure_mode or "None"
            lines.append(f"{p.parameter_value:<18.4f} | {p.metric_value:<10.4f} | {status:<10} | {mode}")

        lines.append("-" * 78)
        lines.append(f"Summary: {self.summary_text}")
        lines.append(border)
        return "\n".join(lines)


def bisection_boundary_search(
    eval_fn: Callable[[float], Tuple[float, bool, Optional[str]]],
    param_name: str,
    low_val: float,
    high_val: float,
    baseline_metric: float,
    tolerance_ratio: float = 0.5,
    max_iterations: int = 15,
    epsilon: float = 1e-3,
) -> PhaseBoundaryReport:
    """Find the exact critical parameter value where performance collapses or stability is lost.
    
    Args:
        eval_fn: Callback taking parameter value and returning (metric, is_stable, failure_mode).
        param_name: Name of parameter being stressed (e.g. 'noise_sigma', 'dt').
        low_val: Lower bound known to be stable.
        high_val: Upper bound suspected or known to break.
        baseline_metric: Nominal metric value.
        tolerance_ratio: Threshold fraction (e.g. 0.5 means breakdown if metric < 0.5 * baseline).
        max_iterations: Maximum bisection steps.
        epsilon: Stopping precision for parameter interval.
        
    Returns:
        PhaseBoundaryReport with critical threshold and evaluated points.
    """
    points: List[StressPoint] = []
    threshold_metric = baseline_metric * tolerance_ratio

    # 1. Evaluate baseline/low
    m_low, s_low, mode_low = eval_fn(low_val)
    points.append(StressPoint(low_val, m_low, s_low, mode_low))

    # 2. Evaluate high
    m_high, s_high, mode_high = eval_fn(high_val)
    points.append(StressPoint(high_val, m_high, s_high, mode_high))

    is_low_ok = s_low and (m_low >= threshold_metric)
    is_high_ok = s_high and (m_high >= threshold_metric)

    if is_low_ok and is_high_ok:
        return PhaseBoundaryReport(
            parameter_name=param_name,
            baseline_param_value=low_val,
            baseline_metric=baseline_metric,
            critical_threshold=None,
            tolerance_ratio=tolerance_ratio,
            points_evaluated=points,
            stable_interval=(low_val, high_val),
            summary_text=f"System remains completely stable across the entire tested interval [{low_val}, {high_val}].",
        )

    if not is_low_ok:
        return PhaseBoundaryReport(
            parameter_name=param_name,
            baseline_param_value=low_val,
            baseline_metric=baseline_metric,
            critical_threshold=low_val,
            tolerance_ratio=tolerance_ratio,
            points_evaluated=points,
            stable_interval=(low_val, low_val),
            summary_text=f"System failed even at minimal parameter value {low_val}.",
        )

    # Bisection search between low (stable) and high (broken)
    stable_bound = low_val
    unstable_bound = high_val

    for _ in range(max_iterations):
        if abs(unstable_bound - stable_bound) <= epsilon:
            break

        mid = (stable_bound + unstable_bound) / 2.0
        m_mid, s_mid, mode_mid = eval_fn(mid)
        points.append(StressPoint(mid, m_mid, s_mid, mode_mid))

        is_mid_ok = s_mid and (m_mid >= threshold_metric)
        if is_mid_ok:
            stable_bound = mid
        else:
            unstable_bound = mid

    critical_val = (stable_bound + unstable_bound) / 2.0
    points.sort(key=lambda p: p.parameter_value)

    return PhaseBoundaryReport(
        parameter_name=param_name,
        baseline_param_value=low_val,
        baseline_metric=baseline_metric,
        critical_threshold=round(critical_val, 4),
        tolerance_ratio=tolerance_ratio,
        points_evaluated=points,
        stable_interval=(low_val, round(stable_bound, 4)),
        summary_text=(
            f"Phase transition identified at {param_name} = {critical_val:.4f}. "
            f"The formulation breaks down when {param_name} exceeds this threshold."
        ),
    )


def sweep_stress_parameter(
    eval_fn: Callable[[float], Tuple[float, bool, Optional[str]]],
    param_name: str,
    min_val: float,
    max_val: float,
    baseline_metric: float,
    steps: int = 10,
    tolerance_ratio: float = 0.5,
) -> PhaseBoundaryReport:
    """Linearly sweep a stress parameter and detect breakdown transition.
    
    Args:
        eval_fn: Evaluation callable returning (metric, is_stable, mode).
        param_name: Name of stress parameter.
        min_val: Starting value.
        max_val: Maximum value.
        baseline_metric: Baseline metric.
        steps: Number of equidistant evaluation points.
        tolerance_ratio: Failure ratio threshold.
        
    Returns:
        PhaseBoundaryReport with swept points and discovered critical threshold.
    """
    if steps < 2:
        steps = 2

    step_size = (max_val - min_val) / (steps - 1)
    points: List[StressPoint] = []
    threshold_metric = baseline_metric * tolerance_ratio

    critical_threshold: Optional[float] = None
    stable_max = min_val

    for i in range(steps):
        val = min_val + i * step_size
        metric, is_stable, mode = eval_fn(val)
        points.append(StressPoint(val, metric, is_stable, mode))

        is_ok = is_stable and (metric >= threshold_metric)
        if is_ok:
            stable_max = val
        elif critical_threshold is None:
            critical_threshold = val

    summary = (
        f"Critical threshold at {param_name} = {critical_threshold:.4f}."
        if critical_threshold is not None
        else "No breakdown detected across swept range."
    )

    return PhaseBoundaryReport(
        parameter_name=param_name,
        baseline_param_value=min_val,
        baseline_metric=baseline_metric,
        critical_threshold=critical_threshold,
        tolerance_ratio=tolerance_ratio,
        points_evaluated=points,
        stable_interval=(min_val, stable_max),
        summary_text=summary,
    )
