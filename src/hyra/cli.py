import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .config import config
from .engine import get_experiment_status, run_experiment_pipeline, start_experiment_detached
from .formal_verifier import verify_lean_proof
from .hypothesis_dag import generate_ablation_variants
from .leakage_detector import analyze_path
from .mechanistic_telemetry import assess_dynamical_system
from .promoter import check_promotion_readiness, promote_to_project
from .run_tracker import format_runs_terminal, load_run_history
from .scaffolding import init_experiment_scaffold
from .skill_distiller import distill_skill
from .skill_exporter import export_skill
from .spec_loader import load_and_validate_spec
from .stress_engine import bisection_boundary_search



def main():
    parser = argparse.ArgumentParser(
        prog="hyra",
        description="HYRA: Autonomous AI Research Harness & Experimentation Platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. init command
    init_parser = subparsers.add_parser("init", help="Initialize a new experiment workspace in staging")
    init_parser.add_argument("exp_id", help="Experiment identifier (e.g., EXP-002)")
    init_parser.add_argument("--title", default="Untitled Experiment", help="Title of the research experiment")
    init_parser.add_argument("--project", default="SpikeNets", help="Target project repository name")

    # 2. run command
    run_parser = subparsers.add_parser("run", help="Run an experiment cycle")
    run_parser.add_argument("exp_id", help="Experiment identifier (e.g., EXP-002)")
    run_parser.add_argument("--background", "-b", action="store_true", help="Launch detached background process")

    # 3. status command
    status_parser = subparsers.add_parser("status", help="Check status, PID liveness, and audit verdict")
    status_parser.add_argument("exp_id", help="Experiment identifier")

    # 4. promote command
    promote_parser = subparsers.add_parser("promote", help="Promote confirmed model to target project on Git branch")
    promote_parser.add_argument("exp_id", help="Experiment identifier")

    # 5. distill command
    distill_parser = subparsers.add_parser("distill", help="Distill verified code into an agent skill (SKILL.md)")
    distill_parser.add_argument("--name", required=True, help="Skill name in lowercase with underscores")
    distill_parser.add_argument("--desc", required=True, help="Short description of skill capability")
    distill_parser.add_argument("--guide", required=True, help="Usage guidelines and theoretical context")
    distill_parser.add_argument("--example", required=True, help="Path to Python reference implementation file")

    # 6. check-leakage command
    check_leakage_parser = subparsers.add_parser(
        "check-leakage", help="Perform static AST audit for data leakage vulnerabilities"
    )
    check_leakage_parser.add_argument("path", help="Path to Python file or directory to inspect")
    check_leakage_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")
    check_leakage_parser.add_argument(
        "--strict", action="store_true", help="Exit with error if any warnings or critical issues are found"
    )

    # 7. verify-proof command
    proof_parser = subparsers.add_parser(
        "verify-proof", help="Formally verify mathematical theorem proof using Lean 4"
    )
    proof_parser.add_argument("path", help="Path to .lean proof file")
    proof_parser.add_argument(
        "--allow-sorry", action="store_true", help="Allow incomplete proofs containing 'sorry'"
    )
    proof_parser.add_argument(
        "--theorems", nargs="*", default=None, help="Names of specific theorems to verify"
    )
    proof_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")

    # 8. export-skill command
    export_parser = subparsers.add_parser(
        "export-skill", help="Export and package an agent skill for Agensi.io, Claude, or Cursor"
    )
    export_parser.add_argument("--name", required=True, help="Skill name identifier (e.g. anti_leakage_researcher)")
    export_parser.add_argument(
        "--target", default="agensi", choices=["agensi", "claude", "cursor"], help="Target agent platform"
    )
    export_parser.add_argument("--output", default="dist", help="Output directory for generated ZIP archive")
    export_parser.add_argument("--desc", default=None, help="Custom skill description")

    # 9. analyze-stability command
    stability_parser = subparsers.add_parser(
        "analyze-stability", help="Perform mechanistic dynamical stability analysis on trajectory data"
    )
    stability_parser.add_argument("path", help="Path to JSON file containing trajectory coordinates")
    stability_parser.add_argument("--name", default="system", help="Identifier of dynamical system")
    stability_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")

    # 10. generate-ablations command
    ablate_parser = subparsers.add_parser(
        "generate-ablations", help="Generate systematic leave-one-out ablation variants"
    )
    ablate_parser.add_argument("exp_id", help="Base experiment ID")
    ablate_parser.add_argument(
        "--components", nargs="+", required=True, help="List of component terms to ablate"
    )
    ablate_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")

    # 11. stress-search command
    stress_parser = subparsers.add_parser(
        "stress-search", help="Adversarial parameter stress search and critical phase boundary detection"
    )
    stress_parser.add_argument("param", help="Parameter name to stress (e.g., noise_sigma, dt)")
    stress_parser.add_argument("--low", type=float, required=True, help="Lower parameter bound (stable baseline)")
    stress_parser.add_argument("--high", type=float, required=True, help="Upper parameter bound (failure suspected)")
    stress_parser.add_argument("--baseline", type=float, required=True, help="Baseline nominal metric value")
    stress_parser.add_argument("--script", required=True, help="Evaluation script returning JSON {metric, is_stable, failure_mode}")
    stress_parser.add_argument("--ratio", type=float, default=0.5, help="Failure tolerance ratio (default 0.5)")
    stress_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON report")

    # 12. mcp command
    subparsers.add_parser("mcp", help="Start the Model Context Protocol stdio server")

    args = parser.parse_args()

    if args.command == "init":
        print(f"Initializing experiment workspace for {args.exp_id}...")
        try:
            res = init_experiment_scaffold(args.exp_id, args.title, args.project)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error initializing experiment: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "run":
        if args.background:
            print(f"Launching detached background execution for {args.exp_id}...")
            res = start_experiment_detached(args.exp_id)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(f"Executing synchronous experiment pipeline for {args.exp_id}...")
            try:
                res = asyncio.run(run_experiment_pipeline(args.exp_id))
                print("Run output:")
                print(res)
            except Exception as e:
                print(f"Execution failed: {e}", file=sys.stderr)
                sys.exit(1)

    elif args.command == "status":
        exp_dir = config.staging_dir / "experiments" / args.exp_id
        if not exp_dir.exists():
            print(f"Experiment {args.exp_id} not found in staging ({exp_dir}).", file=sys.stderr)
            sys.exit(1)

        ready, reason = check_promotion_readiness(exp_dir)
        status_info = get_experiment_status(args.exp_id)
        status_info["promotion_ready"] = ready
        status_info["promotion_reason"] = reason
        print(json.dumps(status_info, indent=2, ensure_ascii=False))

        runs = load_run_history(exp_dir)
        if runs:
            print("\nRun History:")
            print(format_runs_terminal(runs))

    elif args.command == "promote":
        exp_dir = config.staging_dir / "experiments" / args.exp_id
        if not exp_dir.exists():
            print(f"Experiment {args.exp_id} not found in staging ({exp_dir}).", file=sys.stderr)
            sys.exit(1)

        try:
            spec = load_and_validate_spec(exp_dir / "spec.yaml")
            print(f"Promoting {args.exp_id} to production project...")
            res = promote_to_project(spec, exp_dir)
            print("Promotion successful:")
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Promotion failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "distill":
        example_path = Path(args.example)
        if not example_path.is_file():
            print(f"Example code file not found: {example_path}", file=sys.stderr)
            sys.exit(1)

        example_code = example_path.read_text(encoding="utf-8")
        try:
            target_skill = distill_skill(args.name, args.desc, args.guide, example_code)
            print(f"Skill distilled successfully: {target_skill}")
        except Exception as e:
            print(f"Skill distillation failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "check-leakage":
        target = Path(args.path)
        if not target.exists():
            print(f"Target path not found: {target}", file=sys.stderr)
            sys.exit(1)

        try:
            report = analyze_path(target)
            if getattr(args, "json", False):
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(report.format_terminal())

            if not report.passed or (args.strict and report.total_findings > 0):
                sys.exit(1)
        except Exception as e:
            print(f"Anti-leakage audit failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "verify-proof":
        target = Path(args.path)
        try:
            report = verify_lean_proof(
                target,
                disallow_sorry=not getattr(args, "allow_sorry", False),
                target_theorems=getattr(args, "theorems", None),
            )
            if getattr(args, "json", False):
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(report.format_terminal())

            if not report.passed:
                sys.exit(1)
        except Exception as e:
            print(f"Formal verification failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "export-skill":
        try:
            res = export_skill(
                skill_name=args.name,
                target_platform=args.target,
                output_dir=Path(args.output),
                custom_description=args.desc,
            )
            print("Skill exported successfully:")
            print(
                json.dumps(
                    {
                        "skill_name": res.skill_name,
                        "target_platform": res.target_platform,
                        "archive_path": res.archive_path,
                        "files_count": len(res.files_included),
                        "size_bytes": res.size_bytes,
                        "files": res.files_included,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as e:
            print(f"Skill export failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "analyze-stability":
        target = Path(args.path)
        if not target.exists():
            print(f"Error: trajectory file {target} does not exist", file=sys.stderr)
            sys.exit(1)
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                traj_base = data
                traj_pert = None
                jacobian = None
                dt = 0.01
            elif isinstance(data, dict):
                traj_base = data.get("trajectory_base") or data.get("trajectory") or []
                traj_pert = data.get("trajectory_perturbed")
                jacobian = data.get("jacobian")
                dt = float(data.get("dt", 0.01))
            else:
                raise ValueError("JSON must contain an array or object with trajectory data.")

            report = assess_dynamical_system(
                system_name=args.name,
                trajectory_base=traj_base,
                trajectory_perturbed=traj_pert,
                jacobian=jacobian,
                dt=dt,
            )
            if getattr(args, "json", False):
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(report.format_terminal())
        except Exception as e:
            print(f"Stability analysis failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "generate-ablations":
        try:
            variants = generate_ablation_variants(args.exp_id, args.components)
            if getattr(args, "json", False):
                print(json.dumps([v.to_dict() for v in variants], indent=2, ensure_ascii=False))
            else:
                border = "=" * 78
                print(border)
                print(f"HYRA SYSTEMATIC ABLATION MATRIX: {args.exp_id}")
                print(border)
                print(f"{'Variant ID':<25} | {'Type':<15} | {'Active Components'}")
                print("-" * 78)
                for v in variants:
                    comps_str = ", ".join(v.active_components) if v.active_components else "(none)"
                    print(f"{v.variant_id:<25} | {v.variant_type:<15} | {comps_str}")
                print(border)
        except Exception as e:
            print(f"Failed to generate ablation variants: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "stress-search":
        script_path = Path(args.script).resolve()
        if not script_path.exists():
            print(f"Error: Evaluation script {script_path} does not exist", file=sys.stderr)
            sys.exit(1)

        def _eval_runner(val: float):
            cmd = [sys.executable, str(script_path), f"--{args.param}", str(val)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
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
                param_name=args.param,
                low_val=args.low,
                high_val=args.high,
                baseline_metric=args.baseline,
                tolerance_ratio=args.ratio,
            )
            if getattr(args, "json", False):
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(report.format_terminal())
        except Exception as e:
            print(f"Stress search failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()


if __name__ == "__main__":
    main()
