"""Scientific Run Tracker for autonomous experimentation.

Maintains an immutable append-only ledger (`runs.jsonl`) in the experiment directory,
recording execution metadata: Git commit SHA, random seed, peak resource utilization
(RAM/VRAM), formal proof status, anti-leakage audit results, and reviewer verdicts.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunRecord:
    run_id: str
    experiment_id: str
    timestamp_iso: str
    git_commit: Optional[str] = None
    random_seed: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    formal_proof_status: str = "NOT_APPLICABLE"
    leakage_status: str = "NOT_APPLICABLE"
    verdict: str = "PENDING"
    exit_code: Optional[int] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            timestamp_iso=data["timestamp_iso"],
            git_commit=data.get("git_commit"),
            random_seed=data.get("random_seed"),
            metrics=data.get("metrics", {}),
            formal_proof_status=data.get("formal_proof_status", "NOT_APPLICABLE"),
            leakage_status=data.get("leakage_status", "NOT_APPLICABLE"),
            verdict=data.get("verdict", "PENDING"),
            exit_code=data.get("exit_code"),
            custom_metrics=data.get("custom_metrics", {}),
        )


def get_git_commit_sha(target_dir: Optional[Path] = None) -> Optional[str]:
    """Retrieve Git commit SHA from target directory or current working directory."""
    cwd = target_dir or Path.cwd()
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def record_run(exp_dir: Path, record: RunRecord) -> Path:
    """Append a new run record to runs.jsonl in experiment staging directory."""
    dir_path = Path(exp_dir).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    runs_file = dir_path / "runs.jsonl"

    record_json = json.dumps(record.to_dict(), ensure_ascii=False)
    with open(runs_file, "a", encoding="utf-8") as f:
        f.write(record_json + "\n")

    return runs_file


def load_run_history(exp_dir: Path) -> List[RunRecord]:
    """Load all run records from runs.jsonl in chronological order."""
    runs_file = Path(exp_dir).resolve() / "runs.jsonl"
    if not runs_file.is_file():
        return []

    records: List[RunRecord] = []
    with open(runs_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                records.append(RunRecord.from_dict(data))
            except Exception:
                pass
    return records


def summarize_runs(records: List[RunRecord]) -> Dict[str, Any]:
    """Calculate aggregated statistics and metrics over run history."""
    if not records:
        return {
            "total_runs": 0,
            "verdicts": {},
            "mean_duration_sec": 0.0,
            "max_ram_mb": 0.0,
            "seeds_tested": [],
        }

    verdict_counts: Dict[str, int] = {}
    durations: List[float] = []
    rams: List[float] = []
    seeds: List[int] = []

    for r in records:
        verdict_counts[r.verdict] = verdict_counts.get(r.verdict, 0) + 1
        if "cpu_duration_sec" in r.metrics:
            durations.append(r.metrics["cpu_duration_sec"])
        if "ram_peak_mb" in r.metrics:
            rams.append(r.metrics["ram_peak_mb"])
        if r.random_seed is not None and r.random_seed not in seeds:
            seeds.append(r.random_seed)

    mean_duration = sum(durations) / len(durations) if durations else 0.0
    max_ram = max(rams) if rams else 0.0

    return {
        "total_runs": len(records),
        "verdicts": verdict_counts,
        "mean_duration_sec": round(mean_duration, 2),
        "max_ram_mb": round(max_ram, 2),
        "seeds_tested": sorted(seeds),
    }


def format_runs_terminal(records: List[RunRecord]) -> str:
    """Format run history as an aligned terminal report table."""
    if not records:
        return "No runs recorded in runs.jsonl."

    lines: List[str] = []
    lines.append("-" * 78)
    lines.append(f"{'Run ID':<18} | {'Seed':<5} | {'Verdict':<10} | {'RAM (MB)':<8} | {'Time (s)':<8} | {'Proof'}")
    lines.append("-" * 78)

    for r in records[-10:]:  # Display last 10 runs
        seed_str = str(r.random_seed) if r.random_seed is not None else "-"
        ram_str = f"{r.metrics.get('ram_peak_mb', 0.0):.1f}"
        time_str = f"{r.metrics.get('cpu_duration_sec', 0.0):.1f}"
        lines.append(
            f"{r.run_id[:18]:<18} | {seed_str:<5} | {r.verdict:<10} | {ram_str:<8} | {time_str:<8} | {r.formal_proof_status}"
        )

    summary = summarize_runs(records)
    lines.append("-" * 78)
    lines.append(
        f"Summary: {summary['total_runs']} runs | Seeds: {len(summary['seeds_tested'])} | "
        f"Mean Duration: {summary['mean_duration_sec']}s | Peak RAM: {summary['max_ram_mb']} MB"
    )
    return "\n".join(lines)
