from pathlib import Path
from hyra.scaffolding import init_experiment_scaffold
from hyra.config import HyraConfig


def test_init_experiment_scaffold(tmp_path):
    staging = tmp_path / "staging"
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Research_Index.md").write_text("# Research Index\n## 1. Реестр Экспериментов (`Experiments/`)\n", encoding="utf-8")

    cfg = HyraConfig(
        staging_dir=staging,
        obsidian_vault=vault,
        memory_file=None,
        skills_dir=tmp_path / "skills",
        default_project_path=tmp_path / "project",
        llm_model="test",
    )

    res = init_experiment_scaffold("EXP-101", title="Scaffolding Test", project_name="MockProject", cfg=cfg)

    exp_dir = staging / "experiments" / "EXP-101"
    assert exp_dir.is_dir()
    assert (exp_dir / "spec.yaml").is_file()
    assert (exp_dir / "status.json").is_file()
    assert (exp_dir / "src").is_dir()
    assert (exp_dir / "artifacts").is_dir()

    # Obsidian note check
    card = vault / "Experiments" / "EXP-101.md"
    assert card.is_file()
    assert "EXP-101" in card.read_text(encoding="utf-8")
