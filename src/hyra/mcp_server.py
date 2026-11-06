import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import HyraConfig, config as default_config
from .engine import get_experiment_status, run_experiment_pipeline, start_experiment_detached
from .formal_verifier import verify_lean_proof
from .hypothesis_dag import generate_ablation_variants
from .leakage_detector import analyze_path
from .mechanistic_telemetry import assess_dynamical_system
from .pareto import CandidatePoint, Objective, find_pareto_frontier
from .promoter import promote_to_project
from .scaffolding import init_experiment_scaffold
from .skill_distiller import distill_skill
from .spec_loader import load_and_validate_spec
from .stats_gate import verify_statistical_significance
from .stress_engine import bisection_boundary_search

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
    def hyra_init_experiment(exp_id: str, title: str, project_name: str = "my_project") -> str:
        """Initialize a new experiment in staging, creating spec.yaml, Obsidian card, and Memory entity.
        
        Args:
            exp_id: Unique experiment identifier (e.g. 'EXP-002').
            title: Scientific hypothesis / experiment title.
            project_name: Target project repository name (default: my_project).
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

    @server.tool()
    def hyra_analyze_stability(path_or_json: str, system_name: str = "system") -> str:
        """Perform mechanistic dynamical stability analysis (Lyapunov exponent, contraction mapping, boundedness).
        
        Args:
            path_or_json: Path to JSON trajectory file OR raw JSON string containing trajectory array or object.
            system_name: Identifier name for the dynamical system under investigation.
        """
        try:
            p = Path(path_or_json).expanduser()
            if p.is_file():
                raw = p.read_text(encoding="utf-8")
            else:
                raw = path_or_json

            data = json.loads(raw)
            if isinstance(data, list):
                tb = data
                tp = None
                jac = None
                dt = 0.01
            elif isinstance(data, dict):
                tb = data.get("trajectory_base") or data.get("trajectory") or []
                tp = data.get("trajectory_perturbed")
                jac = data.get("jacobian")
                dt = float(data.get("dt", 0.01))
            else:
                return "Error: Input must be a JSON array of trajectory values or an object."

            report = assess_dynamical_system(
                system_name=system_name,
                trajectory_base=tb,
                trajectory_perturbed=tp,
                jacobian=jac,
                dt=dt,
            )
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Stability analysis error: {str(e)}"

    @server.tool()
    def hyra_generate_ablations(exp_id: str, components: str) -> str:
        """Generate systematic 2^k and leave-one-out ablation configurations for causal attribution.
        
        Args:
            exp_id: Base experiment identifier (e.g. 'EXP-003').
            components: Comma-separated list or JSON array of component names (e.g. 'diffusion,threshold,homeostasis').
        """
        try:
            if components.strip().startswith("["):
                comps = json.loads(components)
            else:
                comps = [c.strip() for c in components.split(",") if c.strip()]

            variants = generate_ablation_variants(exp_id, comps)
            return json.dumps([v.to_dict() for v in variants], indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Ablation generation error: {str(e)}"

    @server.tool()
    def hyra_stress_search(
        param_name: str,
        low_val: float,
        high_val: float,
        baseline_metric: float,
        script_path: str,
        tolerance_ratio: float = 0.5,
    ) -> str:
        """Adversarial parameter stress search via bisection to find critical breakdown threshold theta_c.
        
        Args:
            param_name: Name of parameter being stressed (e.g. 'noise_sigma', 'dt').
            low_val: Lower parameter bound known to be stable.
            high_val: Upper parameter bound suspected or known to break.
            baseline_metric: Nominal baseline metric value.
            script_path: Path to evaluation script that takes --<param_name> <val> and outputs JSON {metric, is_stable, failure_mode}.
            tolerance_ratio: Fractional threshold for collapse (default 0.5).
        """
        script = Path(script_path).expanduser().resolve()
        if not script.is_file():
            return f"Error: Evaluation script not found: {script_path}"

        def _eval_runner(val: float):
            cmd = [sys.executable, str(script), f"--{param_name}", str(val)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                return 0.0, False, f"Crash (code {proc.returncode}): {proc.stderr[:80].strip()}"
            try:
                out = json.loads(proc.stdout.strip())
                return float(out.get("metric", 0.0)), bool(out.get("is_stable", True)), out.get("failure_mode")
            except Exception:
                return 0.0, False, "Malformed JSON from eval script"

        try:
            report = bisection_boundary_search(
                eval_fn=_eval_runner,
                param_name=param_name,
                low_val=low_val,
                high_val=high_val,
                baseline_metric=baseline_metric,
                tolerance_ratio=tolerance_ratio,
            )
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Stress search error: {str(e)}"

    @server.tool()
    def hyra_pareto_frontier(candidates_json: str, objectives_str: str) -> str:
        """Compute multi-objective non-dominated Pareto frontier across candidate architectures.
        
        Args:
            candidates_json: JSON string or file path containing list of candidate objects with 'candidate_id' and 'metrics'.
            objectives_str: Comma-separated or space-separated list of objectives in format 'metric:direction' (e.g. 'acc:maximize,lat:minimize').
        """
        try:
            cand_p = Path(candidates_json).expanduser()
            if cand_p.is_file():
                raw_text = cand_p.read_text(encoding="utf-8")
            else:
                raw_text = candidates_json

            data = json.loads(raw_text)
            if isinstance(data, dict) and "candidates" in data:
                raw_cands = data["candidates"]
            elif isinstance(data, list):
                raw_cands = data
            else:
                return "Error: Input must be a JSON array of candidates or an object with 'candidates' key."

            candidates: List[CandidatePoint] = []
            for item in raw_cands:
                cid = item.get("candidate_id") or item.get("id") or item.get("name") or "unknown"
                mets = item.get("metrics") or {k: v for k, v in item.items() if isinstance(v, (int, float))}
                meta = item.get("metadata") or {}
                candidates.append(CandidatePoint(candidate_id=cid, metrics={k: float(v) for k, v in mets.items()}, metadata=meta))

            objs_tokens = [o.strip() for o in objectives_str.replace(",", " ").split() if o.strip()]
            objectives: List[Objective] = []
            for token in objs_tokens:
                if ":" in token:
                    oname, odir = token.split(":", 1)
                    odir_norm = "minimize" if odir.lower() in ("min", "minimize") else "maximize"
                else:
                    oname, odir_norm = token, "maximize"
                objectives.append(Objective(name=oname, direction=odir_norm))

            report = find_pareto_frontier(candidates, objectives)
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Pareto analysis error: {str(e)}"

    @server.tool()
    def hyra_verify_significance(
        candidate_scores: str,
        baseline_scores: str,
        metric_name: str = "metric",
        higher_is_better: bool = True,
        is_paired: bool = True,
        alpha: float = 0.05,
    ) -> str:
        """Verify statistical significance between candidate and baseline runs across multiple seeds.
        
        Args:
            candidate_scores: Comma-separated floats or JSON list of candidate seed evaluations.
            baseline_scores: Comma-separated floats or JSON list of baseline seed evaluations.
            metric_name: Target metric identifier.
            higher_is_better: True if larger metric is superior (accuracy), False if lower is superior (loss).
            is_paired: True to run Paired Student's t-test (matched seeds), False for Welch's t-test.
            alpha: Significance threshold (default 0.05).
        """
        try:
            if candidate_scores.strip().startswith("["):
                cands = [float(x) for x in json.loads(candidate_scores)]
            else:
                cands = [float(x.strip()) for x in candidate_scores.split(",") if x.strip()]

            if baseline_scores.strip().startswith("["):
                bases = [float(x) for x in json.loads(baseline_scores)]
            else:
                bases = [float(x.strip()) for x in baseline_scores.split(",") if x.strip()]

            report = verify_statistical_significance(
                candidate_scores=cands,
                baseline_scores=bases,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                alpha=alpha,
                is_paired=is_paired,
            )
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Statistical significance verification error: {str(e)}"

    return server


def main():
    server = create_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
