"""Unified Skill Exporter for autonomous agent ecosystems and marketplaces.

Packages reusable agent skills into distribution-ready ZIP archives tailored
for Agensi.io marketplace, Claude Code (~/.claude/skills), and Cursor IDE (.cursor/skills).
Bundles:
1. Valid SKILL.md manifest with YAML frontmatter, agent prompt, and protocols.
2. Lightweight verification scripts (AST leakage detector).
3. Declarative specification and Zettelkasten card templates.
4. Target-specific installation and quickstart README.md.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from .config import HyraConfig, config as default_config


@dataclass
class ExportedSkillReport:
    skill_name: str
    target_platform: str
    archive_path: str
    files_included: List[str]
    size_bytes: int


def generate_skill_manifest(
    name: str,
    description: str,
    target_platform: str = "agensi",
    author: str = "Hyra Research Team",
    version: str = "1.0.0",
) -> str:
    """Generate high-standard SKILL.md with valid YAML frontmatter and scientific protocol."""
    manifest_data = {
        "name": name,
        "description": description,
        "version": version,
        "author": author,
        "tools": ["run_command", "view_file", "write_to_file"],
        "tags": ["ai-research", "data-leakage", "mlops", "scientific-discovery", "lean4"],
    }
    frontmatter = yaml.dump(manifest_data, sort_keys=False, allow_unicode=True).strip()

    body = f"""---
{frontmatter}
---

# {name}

## Scientific Research Protocol for Autonomous Agents

You are a Senior AI Research & Verification Engineer equipped with the **Hyra Epistemic Verification Harness**.
Your duty is to conduct rigorous, reproducible, zero-leakage research without hallucinating certainty,
silently altering canonical models, or overfitting on test data.

### Four-Step Autonomous Research Protocol

#### Step 1: Declarative Hypothesis Contract (`spec.yaml`)
Before writing training or evaluation code, formulate a declarative contract in `spec.yaml`:
- **Statement**: precise mathematical or algorithmic hypothesis.
- **Falsification Criteria**: target metric, baseline metric, required minimum gain (e.g. >= 5%).
- **Formal Invariants**: specify Lean 4 verification contract if mathematical proofs are required.
- **Compute Budget**: memory and execution limits.

#### Step 2: Pre-Run AST Anti-Leakage Audit
Execute static code analysis prior to running training or evaluation:
```bash
python scripts/leakage_detector.py <script_or_src_dir> --strict
```
Enforce zero-tolerance for:
1. Fitting scalers/encoders before data splits or on test/validation sets.
2. Lookahead bias: negative shifts (`.shift(-N)`) or centered rolling windows in time-series features.
3. Target leakage: retaining target column in feature matrices.

#### Step 3: Sandboxed Execution & Reproducibility
- All scratch runs, checkpoints, and raw logs must live exclusively in an isolated staging workspace (`~/.research_staging/`).
- Generate a self-contained `reproduce.sh` with fixed random seeds.
- Benchmark across multiple seeds and verify peak RAM/VRAM constraints.

#### Step 4: Machine-Checked Verification Gate
- For mathematical claims: verify Lean 4 proof files without `sorry` (`hyra verify-proof proof.lean`).
- For empirical claims: record results in `runs.jsonl` and verify statistical superiority over optimal baselines.
- Only promote code to the primary repository upon passing independent review.

### Attribution & Open Source
Powered by the **Hyra Autonomous AI Research Platform**:
GitHub Repository: https://github.com/ijstt/hyra-research
"""
    return body.strip() + "\n"


def export_skill(
    skill_name: str,
    target_platform: str = "agensi",
    output_dir: Optional[Path] = None,
    custom_description: Optional[str] = None,
    cfg: Optional[HyraConfig] = None,
) -> ExportedSkillReport:
    """Package and export a validated skill into a distribution ZIP archive.
    
    Args:
        skill_name: Identifier of the skill (e.g. 'anti_leakage_researcher').
        target_platform: Target agent platform ('agensi', 'claude', 'cursor').
        output_dir: Directory where the exported ZIP should be saved.
        custom_description: Optional custom capability summary.
        cfg: Optional HyraConfig instance.
        
    Returns:
        ExportedSkillReport with archive metadata.
    """
    valid_targets = {"agensi", "claude", "cursor"}
    if target_platform.lower() not in valid_targets:
        raise ValueError(f"Invalid target platform '{target_platform}'. Must be one of {valid_targets}")

    target_platform = target_platform.lower()
    active_cfg = cfg or default_config
    out_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = custom_description or (
        "Autonomous AI research harness with static AST anti-leakage audit, "
        "reproducible seed tracking, and formal verification gates."
    )

    archive_filename = f"{skill_name}_{target_platform}.zip"
    archive_path = out_dir / archive_filename

    # Source templates and scripts from package
    pkg_root = Path(__file__).resolve().parent.parent.parent
    leakage_script_src = pkg_root / "src" / "hyra" / "leakage_detector.py"
    spec_template_src = pkg_root / "templates" / "spec_template.yaml"
    card_template_src = pkg_root / "templates" / "experiment_card.md"

    # Also check if existing skill directory exists
    repo_skill_dir = pkg_root / "skills" / skill_name

    with tempfile.TemporaryDirectory() as temp_dir_str:
        staging = Path(temp_dir_str) / skill_name
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "scripts").mkdir(exist_ok=True)
        (staging / "templates").mkdir(exist_ok=True)

        # 1. Generate or copy SKILL.md
        skill_md_path = staging / "SKILL.md"
        if repo_skill_dir.is_dir() and (repo_skill_dir / "SKILL.md").is_file():
            shutil.copy2(repo_skill_dir / "SKILL.md", skill_md_path)
        else:
            skill_md_content = generate_skill_manifest(skill_name, desc, target_platform)
            skill_md_path.write_text(skill_md_content, encoding="utf-8")

        # 2. Bundle verification scripts
        if leakage_script_src.is_file():
            shutil.copy2(leakage_script_src, staging / "scripts" / "leakage_detector.py")
        elif repo_skill_dir.is_dir() and (repo_skill_dir / "scripts" / "leakage_detector.py").is_file():
            shutil.copy2(repo_skill_dir / "scripts" / "leakage_detector.py", staging / "scripts" / "leakage_detector.py")

        # 3. Bundle templates
        if spec_template_src.is_file():
            shutil.copy2(spec_template_src, staging / "templates" / "spec_template.yaml")
        if card_template_src.is_file():
            shutil.copy2(card_template_src, staging / "templates" / "experiment_card.md")

        # 4. Generate README.md tailored for platform
        readme_path = staging / "README.md"
        if target_platform == "claude":
            readme_content = f"""# {skill_name} for Claude Code

## Installation
Extract this archive to your Claude Code skills directory:
```bash
unzip {archive_filename} -d ~/.claude/skills/
```

## Usage
Ask Claude Code:
> "Run an anti-leakage audit on my ML pipeline using the anti_leakage_researcher protocol."

Open Source: https://github.com/ijstt/hyra-research
"""
        elif target_platform == "cursor":
            readme_content = f"""# {skill_name} for Cursor IDE

## Installation
Extract this archive into your project root under `.cursor/skills/`:
```bash
mkdir -p .cursor/skills
unzip {archive_filename} -d .cursor/skills/
```

Open Source: https://github.com/ijstt/hyra-research
"""
        else:  # Agensi.io
            readme_content = f"""# {skill_name} — Agensi.io Official AI Research Skill

Autonomous Anti-Leakage Verification and Hypothesis Scaffolding for Claude Code, Cursor, and Copilot.

## Capabilities
- **Static AST Leakage Linter**: Detects pre-split fits, lookahead bias, and target column leakage.
- **Hypothesis Scaffolding**: Standardized declarative contracts with falsification gates.
- **Reproducibility Guarantee**: Seed tracking and resource profiling.

## 1-Click Installation
- **Claude Code**: extract to `~/.claude/skills/{skill_name}/`
- **Cursor IDE**: extract to `.cursor/skills/{skill_name}/`
- **Standalone CLI**: install via `pip install hyra-research` and run `hyra check-leakage <path>`

Star and contribute on GitHub: https://github.com/ijstt/hyra-research
"""
        readme_path.write_text(readme_content.strip() + "\n", encoding="utf-8")

        # 5. Pack into clean ZIP without OS metadata
        files_included: List[str] = []
        if archive_path.exists():
            archive_path.unlink()

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(staging):
                for f_name in sorted(files):
                    if f_name.startswith(".") or f_name.endswith(".pyc") or f_name == "__pycache__":
                        continue
                    full_p = Path(root) / f_name
                    arc_name = full_p.relative_to(staging.parent).as_posix()
                    zf.write(full_p, arc_name)
                    files_included.append(arc_name)

    size_bytes = archive_path.stat().st_size
    return ExportedSkillReport(
        skill_name=skill_name,
        target_platform=target_platform,
        archive_path=str(archive_path),
        files_included=files_included,
        size_bytes=size_bytes,
    )
