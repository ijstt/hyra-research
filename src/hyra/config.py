import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class HyraConfig:
    staging_dir: Path
    obsidian_vault: Optional[Path]
    memory_file: Optional[Path]
    skills_dir: Path
    default_project_path: Path
    llm_model: str

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "HyraConfig":
        home = Path.home()
        data = {}

        candidate_paths = [
            config_path,
            Path("hyra.yaml"),
            Path("hyra.config.yaml"),
            home / ".config" / "hyra" / "config.yaml",
            home / ".hyra" / "config.yaml",
        ]

        for p in candidate_paths:
            if p and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        loaded = yaml.safe_load(f)
                        if isinstance(loaded, dict):
                            data = loaded
                            break
                except Exception:
                    pass

        # Environment variable overrides take precedence over file config
        staging = os.environ.get("HYRA_STAGING_DIR") or data.get("staging_dir")
        if staging:
            staging_dir = Path(staging).expanduser().resolve()
        else:
            default_staging = home / ".research_staging"
            staging_dir = default_staging if default_staging.exists() else home / ".hyra" / "staging"

        obsidian = os.environ.get("HYRA_OBSIDIAN_VAULT") or data.get("obsidian_vault")
        if obsidian:
            obsidian_vault = Path(obsidian).expanduser().resolve()
        else:
            default_obs = home / "OBSIDIAN" / "Antigravity_Memory" / "Research"
            obsidian_vault = default_obs if default_obs.exists() else None

        mem_file = os.environ.get("HYRA_MEMORY_FILE") or data.get("memory_file")
        if mem_file:
            memory_file = Path(mem_file).expanduser().resolve()
        else:
            default_mem = home / ".gemini" / "antigravity" / "memory.jsonl"
            memory_file = default_mem if default_mem.exists() else None

        skills = os.environ.get("HYRA_SKILLS_DIR") or data.get("skills_dir")
        if skills:
            skills_dir = Path(skills).expanduser().resolve()
        else:
            default_skills = home / ".gemini" / "config" / "skills"
            skills_dir = default_skills if default_skills.exists() else home / ".hyra" / "skills"

        project = os.environ.get("HYRA_DEFAULT_PROJECT") or data.get("default_project_path")
        if project:
            default_project_path = Path(project).expanduser().resolve()
        else:
            default_proj = home / "SpikeNets"
            default_project_path = default_proj if default_proj.exists() else home

        model = (
            os.environ.get("HYRA_LLM_MODEL")
            or data.get("llm_model")
            or "gemini-2.5-flash"
        )

        return cls(
            staging_dir=staging_dir,
            obsidian_vault=obsidian_vault,
            memory_file=memory_file,
            skills_dir=skills_dir,
            default_project_path=default_project_path,
            llm_model=model,
        )

    def ensure_directories(self) -> None:
        (self.staging_dir / "experiments").mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


config = HyraConfig.load()
