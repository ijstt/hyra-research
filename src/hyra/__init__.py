"""HYRA: Autonomous AI Research Harness & Experimentation Platform."""

__version__ = "1.0.0"

from .config import HyraConfig, config
from .engine import (
    get_experiment_status,
    run_experiment_pipeline,
    start_experiment_detached,
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
from .scaffolding import init_experiment_scaffold
from .skill_distiller import distill_skill
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
]
