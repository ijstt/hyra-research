import asyncio
import json
from pathlib import Path
from typing import Optional

from .config import HyraConfig, config as default_config
from .engine import get_experiment_status, run_experiment_pipeline, start_experiment_detached
from .formal_verifier import verify_lean_proof
from .leakage_detector import analyze_path
from .promoter import promote_to_project
from .scaffolding import init_experiment_scaffold
from .skill_distiller import distill_skill
from .spec_loader import load_and_validate_spec

# Support both MCP 2.x (MCPServer) and MCP 1.x (FastMCP)
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as MCPServer
    except ImportError:
        MCPServer = None


def create_mcp_server(name: str = "hyra-research", cfg: Optional[HyraConfig] = None):
    """Factory creating configured Model Context Protocol server."""
    if MCPServer is None:
        raise ImportError(
            "MCP library is not installed. Please install 'mcp' via 'pip install hyra-research[mcp]'."
        )

    active_cfg = cfg or default_config
    server = MCPServer(name)

    @server.tool()
    def hyra_init_experiment(exp_id: str, title: str, project_name: str = "SpikeNets") -> str:
        """Initialize a new experiment in staging, creating spec.yaml, Obsidian card, and Memory entity.
        
        Args:
            exp_id: Unique experiment identifier (e.g. 'EXP-002').
            title: Scientific hypothesis / experiment title.
            project_name: Target project repository name (default: SpikeNets).
        """
        try:
            res = init_experiment_scaffold(exp_id, title, project_name, cfg=active_cfg)
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Initialization error: {str(e)}"

    @server.tool()
    def hyra_start_experiment(exp_id: str, background: bool = True) -> str:
        """Execute autonomous research cycle (Planner -> Worker -> Adversarial Reviewer).
        
        Args:
            exp_id: Experiment ID to execute.
            background: Launch as detached background process (recommended True).
        """
        try:
            if background:
                res = start_experiment_detached(exp_id, cfg=active_cfg)
                return json.dumps(res, indent=2, ensure_ascii=False)
            else:
                res = asyncio.run(run_experiment_pipeline(exp_id, cfg=active_cfg))
                return f"Synchronous run result:\n{res}"
        except Exception as e:
            return f"Execution error: {str(e)}"

    @server.tool()
    def hyra_check_status(exp_id: str) -> str:
        """Check experiment liveness, background PID, latest logs, and audit verdict.
        
        Args:
            exp_id: Experiment ID to inspect.
        """
        try:
            status = get_experiment_status(exp_id, cfg=active_cfg)
            return json.dumps(status, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Status check error: {str(e)}"

    @server.tool()
    def hyra_promote_experiment(exp_id: str) -> str:
        """Promote verified models and notebooks to target Git repository on new branch.
        
        Args:
            exp_id: Confirmed experiment ID to promote.
        """
        staging_root = active_cfg.staging_dir / "experiments"
        exp_dir = staging_root / exp_id
        if not exp_dir.exists():
            return f"Error: Experiment staging directory not found: {exp_dir}"

        try:
            spec = load_and_validate_spec(exp_dir / "spec.yaml")
            res = promote_to_project(spec, exp_dir, cfg=active_cfg)
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Promotion error: {str(e)}"

    @server.tool()
    def hyra_distill_skill(name: str, description: str, guidelines: str, example_code: str) -> str:
        """Distill confirmed algorithmic or computational pattern into permanent SKILL.md.
        
        Args:
            name: Lowercase skill identifier with underscores (e.g., 'jump_diffusion_step').
            description: Short summary of capability for agent activation.
            guidelines: Theoretical invariants and usage rules.
            example_code: Clean reference Python implementation.
        """
        try:
            skill_file = distill_skill(name, description, guidelines, example_code, cfg=active_cfg)
            return f"Skill successfully distilled to: {skill_file}"
        except Exception as e:
            return f"Distillation error: {str(e)}"

    @server.tool()
    def hyra_check_leakage(path: str) -> str:
        """Run AST static analysis on Python script or directory to detect data leakage antipatterns.
        
        Args:
            path: Absolute or relative path to a Python script or folder of scripts.
        """
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"Error: Target path not found: {path}"

        try:
            report = analyze_path(target)
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Leakage check error: {str(e)}"

    @server.tool()
    def hyra_verify_proof(path: str, disallow_sorry: bool = True) -> str:
        """Formally verify Lean 4 mathematical proof file and check for incomplete 'sorry' assertions.
        
        Args:
            path: Path to the .lean proof file.
            disallow_sorry: Reject proofs containing incomplete 'sorry' assertions (default True).
        """
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            return f"Error: Proof file not found: {path}"

        try:
            report = verify_lean_proof(target, disallow_sorry=disallow_sorry)
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Formal proof verification error: {str(e)}"

    return server


def main():
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
