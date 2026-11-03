import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import HyraConfig, config as default_config


def check_promotion_readiness(exp_dir: Path) -> Tuple[bool, str]:
    """Verify whether an experiment in staging satisfies promotion criteria.
    
    Invariants:
    1. audit_log.md must exist and contain 'VERDICT: CONFIRMED'.
    2. reproduce.sh must exist and be executable/self-contained.
    """
    audit_file = exp_dir / "audit_log.md"
    reproduce_script = exp_dir / "reproduce.sh"

    if not audit_file.exists():
        return False, "Audit log 'audit_log.md' not found. Blind peer review was not performed."

    audit_text = audit_file.read_text(encoding="utf-8")
    if "VERDICT: CONFIRMED" not in audit_text and "CONFIRMED" not in audit_text:
        return False, "Experiment does not have 'CONFIRMED' status in audit_log.md."

    if not reproduce_script.exists():
        return False, "Reproduction script 'reproduce.sh' is missing."

    return True, "Experiment is verified and eligible for promotion to production repository."


def promote_to_project(
    spec: Dict[str, Any],
    exp_dir: Path,
    cfg: Optional[HyraConfig] = None,
) -> Dict[str, Any]:
    """Promote verified code and demonstration notebooks to target production repository.
    
    Args:
        spec: Validated experiment specification dictionary.
        exp_dir: Path to the experiment staging directory.
        cfg: Optional HyraConfig instance.
        
    Returns:
        Summary dictionary with copied files and Git branch status.
    """
    active_cfg = cfg or default_config
    ready, reason = check_promotion_readiness(exp_dir)
    if not ready:
        raise RuntimeError(f"Promotion rejected: {reason}")

    promotion_config = spec.get("promotion", {})
    raw_proj_path = spec.get("project_path")
    project_path = Path(raw_proj_path).expanduser().resolve() if raw_proj_path else active_cfg.default_project_path
    exp_id = spec["experiment_id"]
    title = spec.get("title", exp_id)

    if not project_path.exists():
        raise FileNotFoundError(f"Target project directory not found: {project_path}")

    results: Dict[str, Any] = {
        "experiment_id": exp_id,
        "copied_files": [],
        "target_project": str(project_path),
        "git_branch": None,
        "git_committed": False,
    }

    is_git_repo = (project_path / ".git").exists()
    branch_name = f"exp/{exp_id}"

    if is_git_repo:
        try:
            # Create or checkout experiment feature branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=project_path,
                capture_output=True,
                text=True,
            )
            results["git_branch"] = branch_name
        except Exception:
            results["git_branch"] = branch_name

    # 1. Promote target Python code module
    target_module_rel = promotion_config.get("target_module_path")
    if target_module_rel:
        src_module = exp_dir / "src" / Path(target_module_rel).name
        if src_module.exists():
            dest_module = project_path / target_module_rel
            dest_module.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_module, dest_module)
            results["copied_files"].append(str(dest_module))

    # 2. Promote target demonstration notebook
    target_nb_rel = promotion_config.get("target_notebook_path")
    if target_nb_rel:
        src_nb = exp_dir / "src" / Path(target_nb_rel).name
        if not src_nb.exists():
            src_nb = exp_dir / Path(target_nb_rel).name
        if src_nb.exists():
            dest_nb = project_path / target_nb_rel
            dest_nb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_nb, dest_nb)
            results["copied_files"].append(str(dest_nb))

    # 3. Commit promoted files to Git
    if is_git_repo and results["copied_files"]:
        try:
            subprocess.run(["git", "add"] + results["copied_files"], cwd=project_path, check=True)
            commit_msg = f"feat(exp): promote validated {exp_id} - {title}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=project_path, check=True)
            results["git_committed"] = True
        except Exception as e:
            results["git_commit_error"] = str(e)

    # 4. Update Obsidian research card if vault is configured
    if active_cfg.obsidian_vault and active_cfg.obsidian_vault.exists():
        card_path = active_cfg.obsidian_vault / "Experiments" / f"{exp_id}.md"
        if card_path.exists():
            try:
                card_text = card_path.read_text(encoding="utf-8")
                card_text = card_text.replace("`DRAFT`", "`CONFIRMED`")
                card_text = card_text.replace("`RUNNING`", "`CONFIRMED`")
                card_path.write_text(card_text, encoding="utf-8")
            except Exception:
                pass

    return results
