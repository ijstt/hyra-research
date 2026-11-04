import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import HyraConfig, config as default_config


def init_experiment_scaffold(
    exp_id: str,
    title: str = "Untitled Experiment",
    project_name: str = "my_project",
    cfg: Optional[HyraConfig] = None,
) -> Dict[str, Any]:
    """Initialize an isolated experiment workspace in staging with templates and tracking.
    
    Args:
        exp_id: Unique experiment identifier (e.g. 'EXP-002').
        title: Scientific description of the experiment.
        project_name: Target project repository name.
        cfg: Optional HyraConfig instance.
        
    Returns:
        Dictionary containing experiment path and initialization metadata.
    """
    active_cfg = cfg or default_config
    active_cfg.ensure_directories()

    exp_dir = active_cfg.staging_dir / "experiments" / exp_id
    if exp_dir.exists():
        raise FileExistsError(f"Experiment staging directory already exists: {exp_dir}")

    (exp_dir / "src").mkdir(parents=True, exist_ok=True)
    (exp_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Initialize status.json
    status_data = {
        "experiment_id": exp_id,
        "title": title,
        "project_name": project_name,
        "status": "DRAFT",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "pid": None,
    }
    (exp_dir / "status.json").write_text(
        json.dumps(status_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2. Populate spec.yaml from package template
    pkg_template = Path(__file__).resolve().parent.parent.parent / "templates" / "spec_template.yaml"
    target_spec = exp_dir / "spec.yaml"

    if pkg_template.is_file():
        content = pkg_template.read_text(encoding="utf-8")
        content = content.replace('experiment_id: "EXP-001"', f'experiment_id: "{exp_id}"')
        content = content.replace('title: "Название эксперимента"', f'title: "{title}"')
        content = content.replace('date: "2026-11-02"', f'date: "{today_str}"')
        target_spec.write_text(content, encoding="utf-8")
    else:
        # Fallback minimal spec
        minimal_spec = (
            f'experiment_id: "{exp_id}"\n'
            f'title: "{title}"\n'
            f'date: "{today_str}"\n'
            f'hypothesis:\n'
            f'  statement: "Investigate target mathematical formulation"\n'
            f'falsification:\n'
            f'  target_metric: "val_loss"\n'
            f'  baseline_metric: 1.0\n'
            f'  required_gain: 0.05\n'
            f'compute_budget:\n'
            f'  max_runtime_minutes: 60\n'
        )
        target_spec.write_text(minimal_spec, encoding="utf-8")

    # 3. Create Obsidian Zettelkasten card if vault is configured
    obsidian_card_created = False
    if active_cfg.obsidian_vault and active_cfg.obsidian_vault.exists():
        exp_vault = active_cfg.obsidian_vault / "Experiments"
        exp_vault.mkdir(parents=True, exist_ok=True)
        target_card = exp_vault / f"{exp_id}.md"

        card_tpl = Path(__file__).resolve().parent.parent.parent / "templates" / "experiment_card.md"
        if card_tpl.is_file():
            card_content = card_tpl.read_text(encoding="utf-8")
            card_content = card_content.replace("{{EXPERIMENT_ID}}", exp_id)
            card_content = card_content.replace("{{EXPERIMENT_TITLE}}", title)
            card_content = card_content.replace("{{DATE}}", today_str)
            card_content = card_content.replace("{{PROJECT_NAME}}", project_name)
            target_card.write_text(card_content, encoding="utf-8")
        else:
            target_card.write_text(f"# Experiment {exp_id}: {title}\nDate: {today_str}\nStatus: `DRAFT`\n", encoding="utf-8")
        obsidian_card_created = True

        # Update Research_Index.md if present
        index_file = active_cfg.obsidian_vault / "Research_Index.md"
        if index_file.is_file():
            idx_text = index_file.read_text(encoding="utf-8")
            entry = f"- [[{exp_id}|{exp_id}: {title}]] — Status: `DRAFT` (created {today_str})\n"
            if entry not in idx_text:
                marker = "## 1. Реестр Экспериментов (`Experiments/`)\n"
                if marker in idx_text:
                    parts = idx_text.split(marker, 1)
                    idx_text = parts[0] + marker + entry + parts[1]
                else:
                    idx_text += "\n" + entry
                index_file.write_text(idx_text, encoding="utf-8")

    # 4. Register in Knowledge Graph (Memory MCP JSONL) if configured
    memory_logged = False
    if active_cfg.memory_file and active_cfg.memory_file.is_file():
        try:
            entity_record = {
                "type": "entity",
                "name": f"Experiment_{exp_id}",
                "entityType": "Experiment",
                "observations": [
                    f"ID: {exp_id}",
                    f"Title: {title}",
                    f"Target Project: {project_name}",
                    "Status: DRAFT",
                    f"Staging Directory: {exp_dir}",
                    f"Created At: {datetime.now().isoformat()}",
                ],
            }
            relation_record = {
                "type": "relation",
                "from": f"Experiment_{exp_id}",
                "to": "Hyra_Research_Contour",
                "relationType": "PART_OF",
            }
            with open(active_cfg.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entity_record, ensure_ascii=False) + "\n")
                f.write(json.dumps(relation_record, ensure_ascii=False) + "\n")
            memory_logged = True
        except Exception:
            pass

    return {
        "experiment_id": exp_id,
        "title": title,
        "staging_dir": str(exp_dir),
        "spec_file": str(target_spec),
        "obsidian_card": str(active_cfg.obsidian_vault / "Experiments" / f"{exp_id}.md") if obsidian_card_created else None,
        "memory_graph_registered": memory_logged,
    }
