"""Tests for hyra.mechanistic_telemetry module."""

import math
import pytest

from hyra.mechanistic_telemetry import (
    StabilityReport,
    analyze_phase_space_bounds,
    assess_dynamical_system,
    compute_lyapunov_exponent,
    estimate_lipschitz_constant,
    estimate_spectral_radius_matrix,
)


def test_lyapunov_stable_and_divergent():
    # 1. Stable convergent trajectories (x(t) = e^(-0.5 t))
    base_stable = [math.exp(-0.5 * t) for t in range(50)]
    pert_stable = [math.exp(-0.5 * t) + 0.01 * math.exp(-0.5 * t) for t in range(50)]
    lambda_stable = compute_lyapunov_exponent(base_stable, pert_stable, dt=1.0)
    assert lambda_stable < 0.0

    # 2. Divergent trajectories (x(t) = e^(0.5 t))
    base_div = [math.exp(0.2 * t) for t in range(30)]
    pert_div = [math.exp(0.2 * t) + 0.001 * math.exp(0.4 * t) for t in range(30)]
    lambda_div = compute_lyapunov_exponent(base_div, pert_div, dt=1.0)
    assert lambda_div > 0.0


def test_lipschitz_contraction_and_expansion():
    # Contraction mapping f(x) = 0.4 * x
    inputs = [[float(i)] for i in range(10)]
    outputs_contract = [[0.4 * float(i)] for i in range(10)]
    l_contract = estimate_lipschitz_constant(inputs, outputs_contract)
    assert 0.39 <= l_contract <= 0.41

    # Expansive mapping f(x) = 2.5 * x
    outputs_expand = [[2.5 * float(i)] for i in range(10)]
    l_expand = estimate_lipschitz_constant(inputs, outputs_expand)
    assert 2.49 <= l_expand <= 2.51


def test_phase_space_bounds():
    # Bounded sinusoidal trajectory
    sine_traj = [[math.sin(0.1 * t), math.cos(0.1 * t)] for t in range(100)]
    is_bounded, extents, variances = analyze_phase_space_bounds(sine_traj)
    assert is_bounded
    assert len(extents) == 2
    assert extents[0] <= 2.1
    assert variances[0] > 0.0

    # Divergent trajectory
    unbounded_traj = [[10.0 ** i] for i in range(15)]
    is_bounded_div, _, _ = analyze_phase_space_bounds(unbounded_traj, threshold_divergence=1e5)
    assert not is_bounded_div


def test_spectral_radius_matrix():
    # 2x2 matrix with known eigenvalues 0.8 and 0.3
    mat = [
        [0.8, 0.0],
        [0.0, 0.3],
    ]
    rho = estimate_spectral_radius_matrix(mat)
    assert 0.79 <= rho <= 0.81

    # Unstable matrix with dominant eigenvalue 1.5
    mat_unstable = [
        [1.5, 0.2],
        [0.0, 0.7],
    ]
    rho_unstable = estimate_spectral_radius_matrix(mat_unstable)
    assert 1.45 <= rho_unstable <= 1.55


def test_assess_dynamical_system_full():
    # 1. Contractive system
    contractive_traj = [10.0 * (0.5 ** t) for t in range(20)]
    contractive_pert = [10.01 * (0.5 ** t) for t in range(20)]
    report = assess_dynamical_system(
        system_name="contractive_lif",
        trajectory_base=contractive_traj,
        trajectory_perturbed=contractive_pert,
        jacobian=[[0.5]],
    )
    assert report.is_contractive
    assert "STABLE" in report.stability_verdict
    term = report.format_terminal()
    assert "HYRA MECHANISTIC TELEMETRY & STABILITY REPORT" in term

    # 2. Divergent system
    exploding_traj = [10.0 ** i for i in range(10)]
    report_div = assess_dynamical_system(
        system_name="exploding_sde",
        trajectory_base=exploding_traj,
    )
    assert not report_div.is_bounded
    assert "DIVERGENT" in report_div.stability_verdict
