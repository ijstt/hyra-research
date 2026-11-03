from pathlib import Path
from typing import Optional

from .config import HyraConfig, config as default_config


def distill_skill(
    skill_name: str,
    description: str,
    usage_guidelines: str,
    python_example: str,
    destination_dir: Optional[Path] = None,
    cfg: Optional[HyraConfig] = None,
) -> Path:
    """Distill a confirmed algorithmic or engineering technique into a reusable agent skill.
    
    Args:
        skill_name: Unique skill identifier (e.g., 'euler_maruyama_jump_diffusion').
        description: Brief 1-2 sentence description for agent activation.
        usage_guidelines: Usage instructions and theoretical boundaries.
        python_example: Canonical verified Python reference implementation.
        destination_dir: Optional custom target directory for skills.
        cfg: Optional HyraConfig instance.
        
    Returns:
        Path to the generated SKILL.md file.
    """
    active_cfg = cfg or default_config
    target_base = destination_dir or active_cfg.skills_dir
    target_dir = target_base / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)

    skill_file = target_dir / "SKILL.md"
    content = f"""---
name: {skill_name}
description: "{description}"
---

# {skill_name}

## Description
{description}

## Usage Guidelines
{usage_guidelines}

## Canonical Implementation
```python
{python_example.strip()}
```
"""
    skill_file.write_text(content.strip() + "\n", encoding="utf-8")
    return skill_file
