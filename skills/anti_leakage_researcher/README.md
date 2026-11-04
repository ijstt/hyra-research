# Anti-Leakage Verification & Hypothesis Scaffolding

> Official Autonomous AI Research Skill for Claude Code, Cursor, Copilot, and Google Antigravity.
> Developed by Vladimir Fedorko | Powered by [hyra-research](https://github.com/ijstt/hyra-research)

---

## Overview

When LLM agents conduct machine learning or quantitative research autonomously, they frequently fall into subtle data leakage traps:
- Pre-split transformations (fitting scalers and encoders on test or full datasets).
- Lookahead bias (negative `.shift(-N)` or centered rolling windows in time-series features).
- Target leakage (retaining target columns in feature sets or passing full dataframes into splits).
- False positives in ungrounded LaTeX derivations.

**Anti-Leakage Researcher** equips your agent with a deterministic Python AST static analyzer and a four-step scientific research protocol, preventing false discoveries before model training begins.

---

## Key Features

1. **Deterministic AST Static Linter**:
   - `HYRA-LEAK-001` & `002`: Pre-split scaler and transformer fittings.
   - `HYRA-LOOKAHEAD-001` & `002`: Negative shifts and centered window aggregations.
   - `HYRA-LEAK-003`, `004`, `005`: Target column leakage into feature matrices.
2. **Declarative Hypothesis Contracts (`spec.yaml`)**:
   - Strict falsification criteria: target metric, baseline metric, required minimum gain.
   - Formal verification integration for machine-checked Lean 4 theorem proving.
3. **Zero Project Pollution**:
   - Isolates all autonomous research in staging sandboxes (`~/.research_staging/`).
   - Ensures production repositories remain clean with reproducible seeds.

---

## 1-Click Installation

### For Claude Code
Extract the skill archive into your Claude Code skills directory:
```bash
unzip anti_leakage_researcher_claude.zip -d ~/.claude/skills/
```

### For Cursor IDE
Extract the skill archive into your project root under `.cursor/skills/`:
```bash
mkdir -p .cursor/skills
unzip anti_leakage_researcher_cursor.zip -d .cursor/skills/
```

### For CLI & MCP Server
Install the full platform via `uv` or `pip`:
```bash
pip install hyra-research
hyra check-leakage <path_to_code> --strict
```

---

## Agent Usage Example

Prompt your agent:
```
Conduct research on experiment EXP-002 using the anti_leakage_researcher protocol:
1. Define hypothesis and falsification threshold in spec.yaml.
2. Run scripts/leakage_detector.py on all feature pipelines before training.
3. Validate that test datasets remain strictly isolated.
```

---

## Open Source & Community

This skill is a free, open-source module of the **Hyra Autonomous AI Research Platform**.
If you find this skill valuable for your research workflow, please consider starring the repository:

GitHub: https://github.com/ijstt/hyra-research
License: MIT
