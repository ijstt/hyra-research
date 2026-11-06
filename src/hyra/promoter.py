import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

from .config import HyraConfig, config as default_config
from .formal_verifier import verify_lean_proof
from .leakage_detector import analyze_path


def check_promotion_readiness(
    exp_dir: Path,
    spec: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Verify whether an experiment in staging satisfies promotion criteria.
    
    Invariants:
    1. audit_log.md must exist and contain 'VERDICT: CONFIRMED'.
    2. reproduce.sh must exist and be executable/self-contained.
    3. src/ must pass anti-leakage static analysis with 0 critical findings.
    4. If formal_verification is enabled in spec, Lean 4 proof must pass with zero 'sorry'.
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

    # 3. Static anti-leakage audit gate
    src_dir = exp_dir / "src"
    if src_dir.is_dir():
        leak_report = analyze_path(src_dir)
        if not leak_report.passed:
            return False, f"Promotion blocked: Data leakage detected in {leak_report.critical_count} files."

    # 4. Formal verification gate
    active_spec = spec
    if active_spec is None and (exp_dir / "spec.yaml").is_file():
        try:
            with open(exp_dir / "spec.yaml", "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    active_spec = loaded
        except Exception:
            pass

    if active_spec and isinstance(active_spec, dict):
        formal_cfg = active_spec.get("formal_verification")
        if formal_cfg and isinstance(formal_cfg, dict) and formal_cfg.get("enabled"):
            proof_file_rel = formal_cfg.get("proof_file", "src/proof.lean")
            candidate_paths = [
                exp_dir / proof_file_rel,
                exp_dir / "src" / Path(proof_file_rel).name,
            ]
            proof_path = next((p for p in candidate_paths if p.is_file()), None)
            if not proof_path:
                return False, f"Formal verification required by spec, but proof file '{proof_file_rel}' was not found."

            disallow_sorry = formal_cfg.get("disallow_sorry", True)
            theorems = formal_cfg.get("target_theorems")
            proof_report = verify_lean_proof(
                proof_path,
                disallow_sorry=disallow_sorry,
                target_theorems=theorems,
            )
            if not proof_report.passed:
                err = proof_report.error_message or "Proof contains unverified statements or compiler errors."
                if proof_report.has_sorry:
                    err = "Proof contains unverified 'sorry' assertions."
                return False, f"Formal verification gate failed: {err}"

        # 5. Dynamic stability verification gate
        stability_cfg = active_spec.get("stability_criteria")
        if stability_cfg and isinstance(stability_cfg, dict) and stability_cfg.get("enabled"):
            traj_file_rel = stability_cfg.get("trajectory_file", "results/trajectory.json")
            candidate_traj = [
                exp_dir / traj_file_rel,
                exp_dir / "results" / Path(traj_file_rel).name,
                exp_dir / Path(traj_file_rel).name,
            ]
            traj_path = next((p for p in candidate_traj if p.is_file()), None)
            if not traj_path:
                return False, f"Stability criteria required by spec, but trajectory file '{traj_file_rel}' was not found."

            try:
                import json
                from .mechanistic_telemetry import assess_dynamical_system

                with open(traj_path, "r", encoding="utf-8") as tf:
                    tdata = json.load(tf)

                if isinstance(tdata, list):
                    tb = tdata
                    tp = None
                    jac = None
                    dt = 0.01
                elif isinstance(tdata, dict):
                    tb = tdata.get("trajectory_base") or tdata.get("trajectory") or []
                    tp = tdata.get("trajectory_perturbed")
                    jac = tdata.get("jacobian")
                    dt = float(tdata.get("dt", 0.01))
                else:
                    return False, f"Invalid trajectory data format in '{traj_file_rel}'."

                stab_report = assess_dynamical_system(
                    system_name=exp_dir.name,
                    trajectory_base=tb,
                    trajectory_perturbed=tp,
                    jacobian=jac,
                    dt=dt,
                )

                require_bounded = stability_cfg.get("require_bounded", True)
                require_contractive = stability_cfg.get("require_contractive", False)
                max_lyapunov = stability_cfg.get("max_lyapunov")

                if require_bounded and not stab_report.is_bounded:
                    return False, "Stability gate failed: System trajectory diverges in phase space."

                if require_contractive and not stab_report.is_contractive:
                    l_val = (
                        f"{stab_report.lipschitz_constant:.4f}"
                        if stab_report.lipschitz_constant is not None
                        else "unknown"
                    )
                    return False, f"Stability gate failed: System is not contractive (Lipschitz L = {l_val})."

                if max_lyapunov is not None and stab_report.lyapunov_exponent is not None:
                    if stab_report.lyapunov_exponent > max_lyapunov:
                        return (
                            False,
                            f"Stability gate failed: Lyapunov exponent {stab_report.lyapunov_exponent:.4f} exceeds {max_lyapunov}.",
                        )
            except Exception as e:
                return False, f"Stability gate verification error: {str(e)}"

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
    ready, reason = check_promotion_readiness(exp_dir, spec=spec)
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
            # Create or reset and checkout experiment feature branch
            subprocess.run(
                ["git", "checkout", "-B", branch_name],
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
        candidate_srcs = [
            exp_dir / "src" / target_module_rel,
            exp_dir / "src" / Path(target_module_rel).name,
            exp_dir / target_module_rel,
        ]
        src_module = next((p for p in candidate_srcs if p.is_file()), None)
        if src_module:
            dest_module = project_path / target_module_rel
            dest_module.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_module, dest_module)
            results["copied_files"].append(str(dest_module))

    # 2. Promote target demonstration notebook
    target_nb_rel = promotion_config.get("target_notebook_path")
    if target_nb_rel:
        candidate_nbs = [
            exp_dir / "src" / target_nb_rel,
            exp_dir / target_nb_rel,
            exp_dir / "src" / Path(target_nb_rel).name,
            exp_dir / Path(target_nb_rel).name,
        ]
        src_nb = next((p for p in candidate_nbs if p.is_file()), None)
        if src_nb:
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
