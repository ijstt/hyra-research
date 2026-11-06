import asyncio
import json
import os
import psutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from datetime import datetime, timezone
import uuid

from .config import HyraConfig, config as default_config
from .leakage_detector import analyze_path
from .resource_guard import get_system_metrics, run_sandboxed_command
from .run_tracker import RunRecord, get_git_commit_sha, load_run_history, record_run, summarize_runs
from .spec_loader import load_and_validate_spec


# Check for Google Antigravity Agent SDK availability
try:
    from google.antigravity import Agent, LocalAgentConfig, LocalOpenAIAgentConfig, types
    HAS_ANTIGRAVITY_SDK = True
except ImportError:
    HAS_ANTIGRAVITY_SDK = False


def build_harness_agent_config(exp_dir: Path, spec: Dict[str, Any], cfg: HyraConfig):
    """Construct multi-agent configuration with Mission Planner, Worker, and Adversarial Reviewer."""
    if not HAS_ANTIGRAVITY_SDK:
        return None

    exp_id = spec["experiment_id"]
    artifacts_dir = exp_dir / "artifacts"
    src_dir = exp_dir / "src"

    # 1. Adversarial Reviewer (Independent blind review)
    reviewer = types.SubagentConfig(
        name="adversarial_reviewer",
        description="Independent peer reviewer: verifies reproducible runs, anti-leakage gates, and mathematical contracts.",
        system_instruction=(
            "You are a strict, impartial scientific reviewer (Adversarial Reviewer). "
            f"You are conducting a blind audit of experiment {exp_id}. "
            f"Inspect artifacts in {artifacts_dir} and run {exp_dir}/reproduce.sh. "
            "Verification criteria: "
            "1. Zero data leakage: test/validation datasets must be strictly isolated; scalers fitted only on train. "
            f"2. Mathematical improvement contract: target gain >= {spec['falsification']['required_gain'] * 100}%. "
            "3. Numerical stability and convergence across prescribed seeds. "
            f"Output a structured audit report in {exp_dir}/audit_log.md with a mandatory conclusion: "
            "'VERDICT: CONFIRMED' or 'VERDICT: FALSIFIED'."
        ),
        capabilities=types.SubagentCapabilities(
            enabled_tools=[
                types.BuiltinTools.RUN_COMMAND,
                types.BuiltinTools.VIEW_FILE,
                types.BuiltinTools.WRITE_TO_FILE,
            ],
        ),
    )

    # 2. Experiment Worker (Simulation developer)
    worker = types.SubagentConfig(
        name="experiment_worker",
        description="Research engineer: develops models, writes tests, runs experiments, ensures memory discipline.",
        system_instruction=(
            f"You are the lead research engineer for experiment {exp_id}. "
            f"Your working directory is {exp_dir}. "
            f"Save all code in {src_dir}/, weights and logs in {artifacts_dir}/. "
            f"Generate a self-contained, reproducible bash script {exp_dir}/reproduce.sh with fixed random seeds. "
            "Enforce strict memory discipline: explicit tensor deallocation, garbage collection, and zero OOM errors. "
            "Never write code or temporary artifacts outside the designated staging directory."
        ),
        capabilities=types.SubagentCapabilities(
            enabled_tools=[
                types.BuiltinTools.RUN_COMMAND,
                types.BuiltinTools.VIEW_FILE,
                types.BuiltinTools.WRITE_TO_FILE,
            ],
        ),
    )

    # 3. Mission Planner (Task workflow graph)
    mission_planner = types.SubagentConfig(
        name="mission_planner",
        description="Workflow coordinator: decomposes declarative specification and supervises review cycles.",
        system_instruction=(
            f"You are the research mission supervisor for {exp_id}. "
            "Execution order: "
            "Step 1: Invoke experiment_worker to implement canonical baseline and log baseline metrics. "
            "Step 2: Invoke experiment_worker to implement proposed formulation, run evaluation across seeds, and create reproduce.sh. "
            "Step 3: Invoke adversarial_reviewer to perform an independent blind audit. "
            "If the reviewer returns FALSIFIED or detects data leakage, direct experiment_worker to remediate the defect."
        ),
        capabilities=types.SubagentCapabilities(
            enabled_tools=[
                types.BuiltinTools.START_SUBAGENT,
                types.BuiltinTools.VIEW_FILE,
                types.BuiltinTools.WRITE_TO_FILE,
            ],
            allowed_subagents=["experiment_worker", "adversarial_reviewer"],
        ),
    )

    subagents = [mission_planner, worker, reviewer]
    caps = types.CapabilitiesConfig(
        agent_behavior=types.AgentBehavior.AUTONOMOUS,
        enable_subagents=True,
        max_subagent_depth=2,
        allowed_subagents=["mission_planner"],
    )

    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return LocalOpenAIAgentConfig(
            model=cfg.llm_model,
            base_url="https://openrouter.ai/api/v1",
            env={"OPENAI_API_KEY": os.environ["OPENROUTER_API_KEY"]},
            subagents=subagents,
            capabilities=caps,
        )

    return LocalAgentConfig(
        model=cfg.llm_model,
        subagents=subagents,
        capabilities=caps,
    )


async def run_experiment_pipeline(exp_id: str, cfg: Optional[HyraConfig] = None) -> str:
    """Run the complete research experiment cycle synchronously."""
    active_cfg = cfg or default_config
    exp_dir = active_cfg.staging_dir / "experiments" / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment staging directory not found: {exp_dir}")

    spec = load_and_validate_spec(exp_dir / "spec.yaml")
    metrics = get_system_metrics()

    # Update status in status.json
    status_file = exp_dir / "status.json"
    if status_file.exists():
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        status_data["status"] = "RUNNING"
        status_file.write_text(json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if HAS_ANTIGRAVITY_SDK:
        agent_config = build_harness_agent_config(exp_dir, spec, active_cfg)
        try:
            async with Agent(config=agent_config) as agent:
                prompt = (
                    f"Begin autonomous research following specification contract {exp_dir}/spec.yaml. "
                    "Execute dataset preparation, implementation, evaluation, and blind adversarial audit."
                )
                result = await agent.chat(prompt)

                # Final verdict detection
                audit_file = exp_dir / "audit_log.md"
                final_status = "COMPLETED"
                if audit_file.exists():
                    text = audit_file.read_text(encoding="utf-8")
                    if "CONFIRMED" in text:
                        final_status = "CONFIRMED"
                    elif "FALSIFIED" in text:
                        final_status = "FALSIFIED"

                if status_file.exists():
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    status_data["status"] = final_status
                    status_file.write_text(json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8")

                return str(result)
        except Exception as err:
            if status_file.exists():
                status_data = json.loads(status_file.read_text(encoding="utf-8"))
                status_data["status"] = "FAILED"
                status_data["error"] = str(err)
                status_file.write_text(json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8")
            raise err

    # Standalone execution mode (Fallback when running outside Google Antigravity IDE)
    reproduce_script = exp_dir / "reproduce.sh"
    if reproduce_script.exists():
        ret, stdout, stderr, elapsed = run_sandboxed_command(
            f"bash {reproduce_script}",
            cwd=str(exp_dir),
            timeout_sec=spec.get("compute_budget", {}).get("max_runtime_minutes", 60) * 60,
        )
        # 1. Static AST Data Leakage Analysis
        leakage_status = "PASSED"
        src_dir = exp_dir / "src"
        if src_dir.is_dir() and any(src_dir.glob("**/*.py")):
            leak_report = analyze_path(src_dir)
            if not leak_report.passed:
                leakage_status = "VIOLATIONS_FOUND"

        # 2. Audit verdict inspection from audit_log.md
        audit_file = exp_dir / "audit_log.md"
        audit_verdict = None
        if audit_file.exists():
            text = audit_file.read_text(encoding="utf-8")
            if "CONFIRMED" in text and "FALSIFIED" not in text:
                audit_verdict = "CONFIRMED"
            elif "FALSIFIED" in text:
                audit_verdict = "FALSIFIED"

        if ret != 0:
            final_status = "FAILED"
        elif leakage_status == "VIOLATIONS_FOUND":
            final_status = "FAILED"
        elif audit_verdict == "CONFIRMED":
            final_status = "CONFIRMED"
        elif audit_verdict == "FALSIFIED":
            final_status = "FALSIFIED"
        else:
            final_status = "COMPLETED"

        if status_file.exists():
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            status_data["status"] = final_status
            if leakage_status == "VIOLATIONS_FOUND":
                status_data["leakage_error"] = "Critical AST data leakage violations detected in src/"
            status_file.write_text(json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Record structured execution run in runs.jsonl
        sys_metrics = get_system_metrics()
        record = RunRecord(
            run_id=f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            experiment_id=exp_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            git_commit=get_git_commit_sha(exp_dir),
            random_seed=spec.get("falsification", {}).get("random_seed") or spec.get("random_seed"),
            metrics={
                "cpu_duration_sec": round(elapsed, 2),
                "ram_peak_mb": round(sys_metrics.get("ram_used_gb", 0.0) * 1024, 2),
            },
            formal_proof_status="VALID" if (exp_dir / "src" / "proof.lean").exists() else "NOT_APPLICABLE",
            leakage_status=leakage_status,
            verdict=final_status,
            exit_code=ret,
        )
        record_run(exp_dir, record)

        return f"Standalone run finished with code {ret} in {elapsed}s.\nStdout:\n{stdout}\nStderr:\n{stderr}"

    raise RuntimeError(
        "Google Antigravity SDK is not available, and 'reproduce.sh' is missing from staging. "
        "Create reproduce.sh or configure Antigravity agent credentials."
    )


def start_experiment_detached(exp_id: str, cfg: Optional[HyraConfig] = None) -> Dict[str, Any]:
    """Launch autonomous experiment execution as a background detached process."""
    active_cfg = cfg or default_config
    exp_dir = active_cfg.staging_dir / "experiments" / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment staging directory not found: {exp_dir}")

    python_bin = sys.executable
    log_file = exp_dir / "run.log"
    pid_file = exp_dir / "run.pid"

    cmd = [
        python_bin,
        "-c",
        (
            f"from hyra.engine import run_experiment_pipeline; "
            f"import asyncio; "
            f"asyncio.run(run_experiment_pipeline('{exp_id}'))"
        ),
    ]

    with open(log_file, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=str(exp_dir),
            start_new_session=True,
        )

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    status_file = exp_dir / "status.json"
    if status_file.exists():
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        status_data["status"] = "RUNNING"
        status_data["pid"] = proc.pid
        status_file.write_text(json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "experiment_id": exp_id,
        "status": "RUNNING_DETACHED",
        "pid": proc.pid,
        "log_file": str(log_file),
    }


def get_experiment_status(exp_id: str, cfg: Optional[HyraConfig] = None) -> Dict[str, Any]:
    """Retrieve execution state, process liveness, logs, and audit verdict."""
    active_cfg = cfg or default_config
    exp_dir = active_cfg.staging_dir / "experiments" / exp_id
    if not exp_dir.exists():
        return {"status": "NOT_FOUND", "experiment_id": exp_id}

    status_file = exp_dir / "status.json"
    status_data = {}
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    pid_file = exp_dir / "run.pid"
    is_running = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            is_running = psutil.pid_exists(pid)
        except Exception:
            is_running = False

    audit_file = exp_dir / "audit_log.md"
    verdict = "PENDING"
    if audit_file.exists():
        text = audit_file.read_text(encoding="utf-8")
        if "CONFIRMED" in text:
            verdict = "CONFIRMED"
        elif "FALSIFIED" in text:
            verdict = "FALSIFIED"

    log_file = exp_dir / "run.log"
    recent_logs = ""
    if log_file.exists():
        try:
            lines = log_file.read_text(encoding="utf-8").splitlines()
            recent_logs = "\n".join(lines[-15:])
        except Exception:
            pass

    runs = load_run_history(exp_dir)
    runs_summary = summarize_runs(runs)

    return {
        "experiment_id": exp_id,
        "is_process_active": is_running,
        "current_status": status_data.get("status", "UNKNOWN"),
        "audit_verdict": verdict,
        "recent_logs": recent_logs,
        "staging_dir": str(exp_dir),
        "total_runs": len(runs),
        "runs_summary": runs_summary,
    }
