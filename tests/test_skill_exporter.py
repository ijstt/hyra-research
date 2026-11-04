"""Tests for hyra.skill_exporter module and CLI export-skill command."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path
import yaml

import pytest

from hyra.skill_exporter import (
    ExportedSkillReport,
    export_skill,
    generate_skill_manifest,
)


def test_generate_skill_manifest_valid_yaml():
    manifest_str = generate_skill_manifest(
        name="test_skill",
        description="A test capability description",
        target_platform="agensi",
    )
    assert manifest_str.startswith("---\n")
    parts = manifest_str.split("---\n", 2)
    assert len(parts) >= 3
    frontmatter_yaml = parts[1]
    parsed = yaml.safe_load(frontmatter_yaml)
    assert parsed["name"] == "test_skill"
    assert parsed["description"] == "A test capability description"
    assert "tools" in parsed
    assert "run_command" in parsed["tools"]


def test_export_skill_agensi(tmp_path: Path):
    output_dir = tmp_path / "dist"
    report = export_skill(
        skill_name="anti_leakage_researcher",
        target_platform="agensi",
        output_dir=output_dir,
    )

    assert report.skill_name == "anti_leakage_researcher"
    assert report.target_platform == "agensi"
    archive = Path(report.archive_path)
    assert archive.is_file()
    assert archive.stat().st_size > 0

    # Inspect zip contents
    with zipfile.ZipFile(archive, "r") as zf:
        namelist = zf.namelist()
        assert any("SKILL.md" in n for n in namelist)
        assert any("README.md" in n for n in namelist)
        assert any("scripts/leakage_detector.py" in n for n in namelist)
        assert any("templates/spec_template.yaml" in n for n in namelist)
        assert any("templates/experiment_card.md" in n for n in namelist)

        # Ensure no OS junk or pycache
        assert not any("__pycache__" in n for n in namelist)
        assert not any(".DS_Store" in n for n in namelist)

        # Read and validate SKILL.md from zip
        skill_md_name = next(n for n in namelist if n.endswith("SKILL.md"))
        skill_md_bytes = zf.read(skill_md_name).decode("utf-8")
        assert "name: anti_leakage_researcher" in skill_md_bytes


def test_export_skill_claude_and_cursor(tmp_path: Path):
    out_claude = tmp_path / "claude"
    rep_claude = export_skill(
        skill_name="anti_leakage_researcher",
        target_platform="claude",
        output_dir=out_claude,
    )
    assert Path(rep_claude.archive_path).is_file()

    out_cursor = tmp_path / "cursor"
    rep_cursor = export_skill(
        skill_name="anti_leakage_researcher",
        target_platform="cursor",
        output_dir=out_cursor,
    )
    assert Path(rep_cursor.archive_path).is_file()


def test_export_skill_invalid_target(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid target platform"):
        export_skill(
            skill_name="test_skill",
            target_platform="unsupported_platform",
            output_dir=tmp_path,
        )


def test_cli_export_skill(tmp_path: Path):
    out_dir = tmp_path / "cli_dist"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyra.cli",
            "export-skill",
            "--name",
            "anti_leakage_researcher",
            "--target",
            "agensi",
            "--output",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Skill exported successfully" in res.stdout
    expected_zip = out_dir / "anti_leakage_researcher_agensi.zip"
    assert expected_zip.is_file()
