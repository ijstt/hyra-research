"""HYRA: Autonomous AI Research Harness & Experimentation Platform."""

__version__ = "1.0.0"

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
from .leakage_detector import (
    FileLeakageReport,
    LeakageFinding,
    LeakageReport,
    analyze_code,
    analyze_file,
    analyze_path,
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
]
