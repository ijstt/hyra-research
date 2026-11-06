"""HYRA: Autonomous AI Research Harness & Experimentation Platform."""

__version__ = "0.4.0"

from .config import HyraConfig, config
from .engine import (
    get_experiment_status,
    run_experiment_pipeline,
    start_experiment_detached,
)
from .formal_verifier import (
    FormalProofReport,
    LeanDiagnostic,
    is_lean_installed,
    verify_lean_proof,
)
from .hypothesis_dag import (
    AblationMatrixReport,
    AblationVariant,
    CausalComponentAttribution,
    HypothesisDAG,
    HypothesisNode,
    compute_ablation_attributions,
    generate_ablation_variants,
)
from .leakage_detector import (
    FileLeakageReport,
    LeakageFinding,
    LeakageReport,
    analyze_code,
    analyze_file,
    analyze_path,
)
from .mechanistic_telemetry import (
    StabilityReport,
    analyze_phase_space_bounds,
    assess_dynamical_system,
    compute_lyapunov_exponent,
    estimate_lipschitz_constant,
    estimate_spectral_radius_matrix,
)
from .pareto import (
    CandidatePoint,
    Objective,
    ParetoReport,
    compute_pareto_ranking,
    dominates,
    find_pareto_frontier,
)
from .promoter import check_promotion_readiness, promote_to_project
from .resource_guard import get_system_metrics, run_sandboxed_command
from .run_tracker import (
    RunRecord,
    format_runs_terminal,
    get_git_commit_sha,
    load_run_history,
    record_run,
    summarize_runs,
)
from .scaffolding import init_experiment_scaffold
from .skill_distiller import distill_skill
from .skill_exporter import (
    ExportedSkillReport,
    export_skill,
    generate_skill_manifest,
)
from .spec_loader import SpecValidationError, load_and_validate_spec
from .stats_gate import (
    SeedComparisonReport,
    compute_bootstrap_ci,
    compute_cohen_d,
    compute_welch_t_test,
    compute_wilcoxon_signed_rank,
    verify_statistical_significance,
)
from .stress_engine import (
    PhaseBoundaryReport,
    StressPoint,
    bisection_boundary_search,
    sweep_stress_parameter,
)

__all__ = [
    "HyraConfig",
    "config",
    "SpecValidationError",
    "load_and_validate_spec",
    "get_system_metrics",
    "run_sandboxed_command",
    "init_experiment_scaffold",
    "run_experiment_pipeline",
    "start_experiment_detached",
    "get_experiment_status",
    "check_promotion_readiness",
    "promote_to_project",
    "distill_skill",
    "analyze_code",
    "analyze_file",
    "analyze_path",
    "LeakageFinding",
    "FileLeakageReport",
    "LeakageReport",
    "FormalProofReport",
    "LeanDiagnostic",
    "is_lean_installed",
    "verify_lean_proof",
    "RunRecord",
    "record_run",
    "load_run_history",
    "summarize_runs",
    "format_runs_terminal",
    "get_git_commit_sha",
    "ExportedSkillReport",
    "export_skill",
    "generate_skill_manifest",
    "StabilityReport",
    "assess_dynamical_system",
    "compute_lyapunov_exponent",
    "estimate_lipschitz_constant",
    "analyze_phase_space_bounds",
    "estimate_spectral_radius_matrix",
    "HypothesisNode",
    "HypothesisDAG",
    "AblationVariant",
    "CausalComponentAttribution",
    "AblationMatrixReport",
    "generate_ablation_variants",
    "compute_ablation_attributions",
    "StressPoint",
    "PhaseBoundaryReport",
    "bisection_boundary_search",
    "sweep_stress_parameter",
    "Objective",
    "CandidatePoint",
    "ParetoReport",
    "dominates",
    "compute_pareto_ranking",
    "find_pareto_frontier",
    "SeedComparisonReport",
    "compute_cohen_d",
    "compute_bootstrap_ci",
    "compute_welch_t_test",
    "compute_wilcoxon_signed_rank",
    "verify_statistical_significance",
]
