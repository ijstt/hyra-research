import os
from pathlib import Path
from hyra.config import HyraConfig


def test_default_config_loading():
    cfg = HyraConfig.load()
    assert isinstance(cfg.staging_dir, Path)
    assert isinstance(cfg.skills_dir, Path)
    assert isinstance(cfg.llm_model, str)


def test_env_override(monkeypatch, tmp_path):
    custom_staging = tmp_path / "custom_staging"
    custom_skills = tmp_path / "custom_skills"
    monkeypatch.setenv("HYRA_STAGING_DIR", str(custom_staging))
    monkeypatch.setenv("HYRA_SKILLS_DIR", str(custom_skills))
    monkeypatch.setenv("HYRA_LLM_MODEL", "custom-model-4b")

    cfg = HyraConfig.load()
    assert cfg.staging_dir == custom_staging.resolve()
    assert cfg.skills_dir == custom_skills.resolve()
    assert cfg.llm_model == "custom-model-4b"


def test_ensure_directories(tmp_path):
    cfg = HyraConfig(
        staging_dir=tmp_path / "staging",
        obsidian_vault=None,
        memory_file=None,
        skills_dir=tmp_path / "skills",
        default_project_path=tmp_path / "project",
        llm_model="test-model",
    )
    cfg.ensure_directories()
    assert (tmp_path / "staging" / "experiments").is_dir()
    assert (tmp_path / "skills").is_dir()
