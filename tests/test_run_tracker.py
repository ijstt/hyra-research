"""Tests for hyra.run_tracker module and experiment history tracking."""

import json
from pathlib import Path

import pytest

from hyra.run_tracker import (
    RunRecord,
    format_runs_terminal,
    get_git_commit_sha,
    load_run_history,
    record_run,
    summarize_runs,
)


def test_record_and_load_runs(tmp_path: Path):
    exp_dir = tmp_path / "EXP-TRACK"
    
    # 1. Record initial run
    record1 = RunRecord(
        run_id="run-001",
        experiment_id="EXP-TRACK",
        timestamp_iso="2026-11-04T12:00:00Z",
        git_commit="abcdef123456",
        random_seed=42,
        metrics={"cpu_duration_sec": 10.5, "ram_peak_mb": 512.0},
        formal_proof_status="VALID",
        leakage_status="PASSED",
        verdict="CONFIRMED",
        exit_code=0,
    )
    runs_file = record_run(exp_dir, record1)
    assert runs_file.is_file()

    # 2. Record second run with different seed
    record2 = RunRecord(
        run_id="run-002",
        experiment_id="EXP-TRACK",
        timestamp_iso="2026-11-04T12:30:00Z",
        git_commit="abcdef123456",
        random_seed=43,
        metrics={"cpu_duration_sec": 12.5, "ram_peak_mb": 600.0},
        formal_proof_status="VALID",
        leakage_status="PASSED",
        verdict="CONFIRMED",
        exit_code=0,
    )
    record_run(exp_dir, record2)

    # 3. Load history
    history = load_run_history(exp_dir)
    assert len(history) == 2
    assert history[0].run_id == "run-001"
    assert history[0].random_seed == 42
    assert history[1].run_id == "run-002"
    assert history[1].random_seed == 43


def test_summarize_runs():
    records = [
        RunRecord(
            run_id="r1",
            experiment_id="E1",
            timestamp_iso="",
            random_seed=10,
            metrics={"cpu_duration_sec": 10.0, "ram_peak_mb": 200.0},
            verdict="CONFIRMED",
        ),
        RunRecord(
            run_id="r2",
            experiment_id="E1",
            timestamp_iso="",
            random_seed=20,
            metrics={"cpu_duration_sec": 20.0, "ram_peak_mb": 400.0},
            verdict="FALSIFIED",
        ),
    ]

    summary = summarize_runs(records)
    assert summary["total_runs"] == 2
    assert summary["mean_duration_sec"] == 15.0
    assert summary["max_ram_mb"] == 400.0
    assert summary["seeds_tested"] == [10, 20]
    assert summary["verdicts"] == {"CONFIRMED": 1, "FALSIFIED": 1}

    # Empty records
    empty_summary = summarize_runs([])
    assert empty_summary["total_runs"] == 0


def test_format_runs_terminal():
    records = [
        RunRecord(
            run_id="run-20261104-01",
            experiment_id="EXP-01",
            timestamp_iso="",
            random_seed=42,
            metrics={"cpu_duration_sec": 5.2, "ram_peak_mb": 312.4},
            formal_proof_status="VALID",
            verdict="CONFIRMED",
        )
    ]
    table = format_runs_terminal(records)
    assert "Run ID" in table
    assert "42" in table
    assert "CONFIRMED" in table
    assert "VALID" in table
    assert "Summary: 1 runs" in table

    # Empty runs
    assert "No runs recorded" in format_runs_terminal([])


def test_corrupt_runs_jsonl_tolerance(tmp_path: Path):
    exp_dir = tmp_path / "EXP-CORRUPT"
    exp_dir.mkdir()
    runs_file = exp_dir / "runs.jsonl"
    runs_file.write_text(
        '{"run_id": "valid-1", "experiment_id": "E", "timestamp_iso": ""}\n'
        'invalid json line here\n'
        '\n'
        '{"run_id": "valid-2", "experiment_id": "E", "timestamp_iso": ""}\n',
        encoding="utf-8",
    )

    history = load_run_history(exp_dir)
    assert len(history) == 2
    assert history[0].run_id == "valid-1"
    assert history[1].run_id == "valid-2"


def test_get_git_commit_sha():
    sha = get_git_commit_sha()
    assert sha is not None
    assert len(sha) == 40
