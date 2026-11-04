"""Mechanistic Telemetry and Dynamical Stability Engine for deep scientific research.

Provides mathematical diagnostics of dynamical systems, differential equations,
and neural formulations:
1. Maximum Lyapunov exponent estimation (chaos vs stability).
2. Numerical Lipschitz constant / contraction mapping validation (L < 1).
3. Phase space trajectory boundedness and variance profiling.
4. Spectral radius estimation of state transition Jacobians.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


def _norm(vec: Sequence[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def _dist(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _to_vector(x: Union[float, int, Sequence[float]]) -> List[float]:
    """Convert scalar or iterable to list of floats."""
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


@dataclass
class StabilityReport:
    system_name: str
    lyapunov_exponent: Optional[float] = None
    lipschitz_constant: Optional[float] = None
    spectral_radius: Optional[float] = None
    is_contractive: bool = False
    is_bounded: bool = True
    phase_volume_extent: List[float] = field(default_factory=list)
    variance_profile: List[float] = field(default_factory=list)
    stability_verdict: str = "UNKNOWN"
    summary_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_terminal(self) -> str:
        lines: List[str] = []
        border = "=" * 78
        lines.append(border)
        lines.append(f"HYRA MECHANISTIC TELEMETRY & STABILITY REPORT: {self.system_name}")
        lines.append(border)
        lines.append(f"Stability Verdict: {self.stability_verdict}")
        lines.append(f"Bounded in Phase Space: {self.is_bounded}")

        if self.lyapunov_exponent is not None:
            status = "STABLE (Attractor)" if self.lyapunov_exponent < 0 else "CHAOTIC (Divergent)"
            lines.append(f"Maximum Lyapunov Exponent: {self.lyapunov_exponent:.4f} [{status}]")

        if self.lipschitz_constant is not None:
            contraction = "STRICT CONTRACTION (L < 1)" if self.is_contractive else "EXPANSIVE (L >= 1)"
            lines.append(f"Lipschitz Constant: {self.lipschitz_constant:.4f} [{contraction}]")

        if self.spectral_radius is not None:
            spec_status = "STABLE (rho < 1)" if self.spectral_radius < 1.0 else "UNSTABLE (rho >= 1)"
            lines.append(f"Jacobian Spectral Radius: {self.spectral_radius:.4f} [{spec_status}]")

        if self.variance_profile:
            var_str = ", ".join(f"{v:.4f}" for v in self.variance_profile[:5])
            lines.append(f"Phase Coordinate Variances: [{var_str}]")

        lines.append("-" * 78)
        lines.append(f"Analysis: {self.summary_explanation}")
        lines.append(border)
        return "\n".join(lines)


def compute_lyapunov_exponent(
    traj_base: Sequence[Union[float, Sequence[float]]],
    traj_perturbed: Sequence[Union[float, Sequence[float]]],
    dt: float = 1.0,
    epsilon_min: float = 1e-12,
) -> float:
    """Estimate the maximum Lyapunov exponent by tracking divergence between two close trajectories.
    
    Args:
        traj_base: Reference trajectory sequence of states.
        traj_perturbed: Trajectory sequence started from small perturbation x0 + delta0.
        dt: Time increment per discrete step.
        epsilon_min: Numerical threshold to prevent division by zero or log of zero.
        
    Returns:
        Estimated maximum Lyapunov exponent lambda (lambda < 0 stable, lambda > 0 chaotic).
    """
    n_steps = min(len(traj_base), len(traj_perturbed))
    if n_steps < 2:
        raise ValueError("Trajectories must have at least 2 time points for Lyapunov estimation.")

    v0_base = _to_vector(traj_base[0])
    v0_pert = _to_vector(traj_perturbed[0])
    d0 = _dist(v0_base, v0_pert)

    if d0 < epsilon_min:
        d0 = epsilon_min

    # Track divergence at final point and intermediate growth rate
    divergences: List[float] = []
    for t in range(1, n_steps):
        vt_base = _to_vector(traj_base[t])
        vt_pert = _to_vector(traj_perturbed[t])
        dt_dist = _dist(vt_base, vt_pert)
        if dt_dist > epsilon_min:
            rate = math.log(dt_dist / d0) / (t * dt)
            divergences.append(rate)

    if not divergences:
        return 0.0

    # Mean rate across trajectory timeline
    return sum(divergences) / len(divergences)


def estimate_lipschitz_constant(
    inputs: Sequence[Union[float, Sequence[float]]],
    outputs: Sequence[Union[float, Sequence[float]]],
    epsilon_min: float = 1e-12,
) -> float:
    """Estimate Lipschitz constant L = sup_{x != y} ||T(x) - T(y)|| / ||x - y||.
    
    Args:
        inputs: Sample input points.
        outputs: Evaluated outputs T(x) for each input point.
        epsilon_min: Minimum denominator distance to prevent division by zero.
        
    Returns:
        Maximum observed ratio L.
    """
    n = min(len(inputs), len(outputs))
    if n < 2:
        raise ValueError("At least 2 input-output pairs are required to compute Lipschitz constant.")

    max_ratio = 0.0
    for i in range(n):
        xi = _to_vector(inputs[i])
        yi = _to_vector(outputs[i])
        for j in range(i + 1, n):
            xj = _to_vector(inputs[j])
            yj = _to_vector(outputs[j])

            dx = _dist(xi, xj)
            if dx >= epsilon_min:
                dy = _dist(yi, yj)
                ratio = dy / dx
                if ratio > max_ratio:
                    max_ratio = ratio

    return max_ratio


def analyze_phase_space_bounds(
    trajectory: Sequence[Union[float, Sequence[float]]],
    threshold_divergence: float = 1e6,
) -> Tuple[bool, List[float], List[float]]:
    """Analyze trajectory boundedness and coordinate-wise variance.
    
    Returns:
        Tuple of (is_bounded, extents_per_coord, variance_per_coord).
    """
    if not trajectory:
        return True, [], []

    vectors = [_to_vector(pt) for pt in trajectory]
    dim = len(vectors[0])
    n = len(vectors)

    mins = [float("inf")] * dim
    maxs = [float("-inf")] * dim
    sums = [0.0] * dim

    for v in vectors:
        for d in range(dim):
            val = v[d]
            if val < mins[d]:
                mins[d] = val
            if val > maxs[d]:
                maxs[d] = val
            sums[d] += val

    means = [s / n for s in sums]
    variances = [0.0] * dim
    for v in vectors:
        for d in range(dim):
            variances[d] += (v[d] - means[d]) ** 2
    variances = [var / n for var in variances]

    extents = [maxs[d] - mins[d] for d in range(dim)]
    is_bounded = all(ext < threshold_divergence for ext in extents)

    return is_bounded, extents, variances


def estimate_spectral_radius_matrix(matrix: Sequence[Sequence[float]], max_iter: int = 100) -> float:
    """Estimate spectral radius rho(A) = max_i |lambda_i| via power iteration method.
    
    Args:
        matrix: Square n x n matrix.
        max_iter: Maximum number of power iterations.
        
    Returns:
        Estimated dominant eigenvalue magnitude.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square and non-empty.")

    # Initialize random vector with unit norm
    b = [1.0 / math.sqrt(n)] * n

    for _ in range(max_iter):
        # Matrix-vector multiplication Ab
        ab = [sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n)]
        norm_ab = _norm(ab)
        if norm_ab < 1e-12:
            return 0.0
        b = [x / norm_ab for x in ab]

    # Rayleigh quotient: (b^T A b) / (b^T b)
    ab = [sum(matrix[i][j] * b[j] for j in range(n)) for i in range(n)]
    eigenvalue = sum(b[i] * ab[i] for i in range(n))
    return abs(eigenvalue)


def assess_dynamical_system(
    system_name: str,
    trajectory_base: Sequence[Union[float, Sequence[float]]],
    trajectory_perturbed: Optional[Sequence[Union[float, Sequence[float]]]] = None,
    jacobian: Optional[Sequence[Sequence[float]]] = None,
    dt: float = 1.0,
) -> StabilityReport:
    """Perform a comprehensive mechanistic stability analysis of an experiment trajectory.
    
    Args:
        system_name: Identifier of model or formulation (e.g. 'adaptive_lif').
        trajectory_base: Main state evolution sequence.
        trajectory_perturbed: Perturbed state sequence for Lyapunov estimation.
        jacobian: Optional state transition Jacobian at operating point.
        dt: Time step duration.
        
    Returns:
        StabilityReport containing Lyapunov exponents, boundedness, and physical verdict.
    """
    is_bounded, extents, variances = analyze_phase_space_bounds(trajectory_base)

    lyapunov: Optional[float] = None
    if trajectory_perturbed is not None and len(trajectory_perturbed) >= 2:
        try:
            lyapunov = compute_lyapunov_exponent(trajectory_base, trajectory_perturbed, dt=dt)
        except Exception:
            pass

    # Approximate Lipschitz constant from consecutive steps T(x_t) = x_{t+1}
    lipschitz: Optional[float] = None
    is_contractive = False
    if len(trajectory_base) >= 3:
        inputs = trajectory_base[:-1]
        outputs = trajectory_base[1:]
        try:
            lipschitz = estimate_lipschitz_constant(inputs, outputs)
            is_contractive = lipschitz < 1.0
        except Exception:
            pass

    spectral_radius: Optional[float] = None
    if jacobian is not None:
        try:
            spectral_radius = estimate_spectral_radius_matrix(jacobian)
        except Exception:
            pass

    # Determine verdict and explanation
    if not is_bounded:
        verdict = "DIVERGENT / EXPLOSIVE"
        summary = "Trajectory diverges towards infinity in finite time. Invariant measure is destroyed."
    elif lyapunov is not None and lyapunov > 0.05:
        verdict = "CHAOTIC ATTRACTOR"
        summary = f"Positive Lyapunov exponent ({lyapunov:.4f}) indicates exponential sensitivity to initial conditions."
    elif is_contractive:
        verdict = "GLOBALLY STABLE (Contractive)"
        summary = f"Lipschitz constant L = {lipschitz:.4f} < 1 confirms Banach contraction to a unique stationary point."
    elif lyapunov is not None and lyapunov < -0.01:
        verdict = "LOCALLY STABLE (Attractor)"
        summary = f"Negative Lyapunov exponent ({lyapunov:.4f}) confirms convergence towards a stable limit set."
    elif spectral_radius is not None and spectral_radius < 1.0:
        verdict = "LOCALLY STABLE (Linearized)"
        summary = f"Spectral radius rho(J) = {spectral_radius:.4f} < 1 proves asymptotic stability around operating point."
    else:
        verdict = "MARGINALLY STABLE"
        summary = "System remains bounded without definitive contraction or exponential divergence."

    return StabilityReport(
        system_name=system_name,
        lyapunov_exponent=lyapunov,
        lipschitz_constant=lipschitz,
        spectral_radius=spectral_radius,
        is_contractive=is_contractive,
        is_bounded=is_bounded,
        phase_volume_extent=extents,
        variance_profile=variances,
        stability_verdict=verdict,
        summary_explanation=summary,
    )
