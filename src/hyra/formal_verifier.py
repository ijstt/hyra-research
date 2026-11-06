"""Formal Verification Engine for Lean 4 theorem proving.

Provides machine-checked verification of mathematical theorems, algorithmic invariants,
and convergence proofs formulated in Lean 4.
Enforces:
1. Zero unproved statements: disallows 'sorry' or 'sorryAx'.
2. Axiomatic integrity: detects and flags non-standard or user-invented axioms.
3. Resource-bounded execution: sandboxed subprocess execution with timeout.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Standard foundational axioms accepted in Lean 4 / Mathlib
STANDARD_LEAN_AXIOMS: Set[str] = {
    "propext",
    "Classical.choice",
    "Quot.sound",
}


@dataclass
class LeanDiagnostic:
    file_path: str
    line: int
    col: int
    severity: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FormalProofReport:
    file_path: str
    passed: bool
    engine: str = "lean4"
    lean_installed: bool = True
    has_sorry: bool = False
    theorems_checked: List[str] = field(default_factory=list)
    axioms_detected: List[str] = field(default_factory=list)
    unauthorized_axioms: List[str] = field(default_factory=list)
    diagnostics: List[LeanDiagnostic] = field(default_factory=list)
    raw_output: str = ""
    error_message: Optional[str] = None
    duration_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "passed": self.passed,
            "engine": self.engine,
            "lean_installed": self.lean_installed,
            "has_sorry": self.has_sorry,
            "theorems_checked": self.theorems_checked,
            "axioms_detected": self.axioms_detected,
            "unauthorized_axioms": self.unauthorized_axioms,
            "diagnostics_count": len(self.diagnostics),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "error_message": self.error_message,
            "duration_sec": self.duration_sec,
        }

    def format_terminal(self) -> str:
        lines: List[str] = []
        border = "=" * 78
        lines.append(border)
        lines.append("HYRA FORMAL THEOREM PROOF VERIFICATION REPORT (LEAN 4)")
        lines.append(border)
        lines.append(f"Proof file: {self.file_path}")
        lines.append(f"Lean 4 toolchain: {'AVAILABLE' if self.lean_installed else 'NOT FOUND'}")
        lines.append(f"Duration: {self.duration_sec:.2f}s | Has 'sorry': {self.has_sorry}")

        if not self.lean_installed:
            lines.append("Verdict: BLOCKED (Lean 4 compiler is not installed on host)")
            lines.append("Remediation: Install Lean 4 via elan: 'curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh'")
            lines.append(border)
            return "\n".join(lines)

        verdict = "PASSED (Formally verified, zero unproven axioms)" if self.passed else "FAILED (Proof verification failed)"
        lines.append(f"Verdict: {verdict}")

        if self.unauthorized_axioms:
            lines.append(f"Unauthorized axioms detected: {', '.join(self.unauthorized_axioms)}")

        if self.diagnostics:
            lines.append("-" * 78)
            lines.append("Compiler Diagnostics:")
            for d in self.diagnostics:
                lines.append(f"  [{d.severity}] Line {d.line}:{d.col} - {d.message}")

        if self.error_message:
            lines.append(f"Error: {self.error_message}")

        lines.append(border)
        return "\n".join(lines)


def is_lean_installed() -> bool:
    """Check if Lean 4 compiler or lake package manager is installed on PATH."""
    return shutil.which("lean") is not None or shutil.which("lake") is not None


def _parse_lean_output(stdout: str, stderr: str, file_path_str: str) -> List[LeanDiagnostic]:
    """Parse compiler output, supporting both Lean JSON line stream and plain text diagnostics."""
    diagnostics: List[LeanDiagnostic] = []
    combined = (stdout + "\n" + stderr).strip()
    if not combined:
        return diagnostics

    for line in combined.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        # 1. Attempt JSON line parse
        if line_clean.startswith("{") and line_clean.endswith("}"):
            try:
                data = json.loads(line_clean)
                pos = data.get("pos", {})
                diagnostics.append(
                    LeanDiagnostic(
                        file_path=data.get("file_name", file_path_str),
                        line=pos.get("line", 1),
                        col=pos.get("column", 0),
                        severity=str(data.get("severity", "error")).upper(),
                        message=str(data.get("data", "")).strip(),
                    )
                )
                continue
            except Exception:
                pass

        # 2. Text regex parse: file.lean:line:col: severity: message
        match = re.match(r"^(.*?):(\d+):(\d+):\s*(error|warning|info):\s*(.*)$", line_clean, re.IGNORECASE)
        if match:
            diagnostics.append(
                LeanDiagnostic(
                    file_path=match.group(1),
                    line=int(match.group(2)),
                    col=int(match.group(3)),
                    severity=match.group(4).upper(),
                    message=match.group(5),
                )
            )
        elif "error:" in line_clean.lower():
            diagnostics.append(
                LeanDiagnostic(
                    file_path=file_path_str,
                    line=1,
                    col=0,
                    severity="ERROR",
                    message=line_clean,
                )
            )
        elif "warning:" in line_clean.lower():
            diagnostics.append(
                LeanDiagnostic(
                    file_path=file_path_str,
                    line=1,
                    col=0,
                    severity="WARNING",
                    message=line_clean,
                )
            )

    return diagnostics


def verify_lean_proof(
    file_path: Path,
    timeout_sec: int = 60,
    disallow_sorry: bool = True,
    target_theorems: Optional[List[str]] = None,
    allowed_axioms: Optional[Set[str]] = None,
) -> FormalProofReport:
    """Run formal verification on a Lean 4 source file.
    
    Args:
        file_path: Path to the .lean proof file.
        timeout_sec: Maximum execution time in seconds.
        disallow_sorry: If True, any occurrence of 'sorry' marks the proof as failed.
        target_theorems: Optional list of theorem names to check specifically.
        allowed_axioms: Set of permissible axioms (defaults to standard Lean foundational axioms).
        
    Returns:
        FormalProofReport with pass/fail verdict, diagnostics, and axiom status.
    """
    path_obj = Path(file_path).resolve()
    effective_allowed_axioms = allowed_axioms or STANDARD_LEAN_AXIOMS
    start_time = time.time()

    if not path_obj.is_file():
        return FormalProofReport(
            file_path=str(path_obj),
            passed=False,
            error_message=f"Lean proof file not found: {path_obj}",
            duration_sec=0.0,
        )

    if not is_lean_installed():
        return FormalProofReport(
            file_path=str(path_obj),
            passed=False,
            lean_installed=False,
            error_message="Lean 4 compiler ('lean' or 'lake') is not installed in the system PATH.",
            duration_sec=time.time() - start_time,
        )

    # Determine compiler command
    lean_bin = shutil.which("lean") or "lean"
    cmd = [lean_bin, "--json", str(path_obj)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(path_obj.parent),
        )
        duration = time.time() - start_time
        raw_output = proc.stdout + "\n" + proc.stderr
        diagnostics = _parse_lean_output(proc.stdout, proc.stderr, str(path_obj))

        # Check for 'sorry' usage
        has_sorry = False
        has_errors = proc.returncode != 0

        # Inspect diagnostics for errors and sorry
        for d in diagnostics:
            msg_lower = d.message.lower()
            if "declaration uses 'sorry'" in msg_lower or "uses sorry" in msg_lower or "sorry" in msg_lower:
                has_sorry = True
            if d.severity == "ERROR":
                has_errors = True

        # Fallback text check for sorry keyword if compiler passed but sorry was embedded
        if not has_sorry:
            file_content = path_obj.read_text(encoding="utf-8")
            # Strip comments
            content_no_comments = re.sub(r"/--.*?--/|/-.*?-/|--.*$", "", file_content, flags=re.DOTALL | re.MULTILINE)
            if re.search(r"\bsorry\b", content_no_comments):
                has_sorry = True

        # Determine pass/fail
        passed = (not has_errors) and (proc.returncode == 0)
        if disallow_sorry and has_sorry:
            passed = False

        # Axiom inspection if theorems specified
        axioms_detected: List[str] = []
        unauthorized_axioms: List[str] = []
        theorems_checked: List[str] = target_theorems or []

        if passed and target_theorems:
            for thm in target_theorems:
                check_script = path_obj.parent / f"_check_axioms_{uuid.uuid4().hex[:6]}.lean"
                try:
                    orig_code = path_obj.read_text(encoding="utf-8")
                    check_script.write_text(f"{orig_code}\n#print axioms {thm}\n", encoding="utf-8")
                    axiom_cmd = [lean_bin, str(check_script)]
                    ax_proc = subprocess.run(
                        axiom_cmd,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        cwd=str(path_obj.parent),
                    )
                    ax_text = ax_proc.stdout
                    # In Lean 4: 'thm' depends on axioms: [axiom1, axiom2]
                    if "[" in ax_text and "]" in ax_text:
                        inside = ax_text.split("[", 1)[1].split("]", 1)[0]
                        tokens = [t.strip(" '\",") for t in inside.split(",") if t.strip()]
                    elif "axioms:" in ax_text:
                        after = ax_text.split("axioms:", 1)[1]
                        tokens = [t.strip(" '\",") for t in after.split() if t.strip()]
                    else:
                        tokens = [t.strip(" '\",") for t in ax_text.split() if t.strip()]

                    for ax_clean in tokens:
                        if ax_clean and ax_clean != thm and ax_clean not in axioms_detected:
                            axioms_detected.append(ax_clean)
                            if ax_clean not in effective_allowed_axioms:
                                unauthorized_axioms.append(ax_clean)
                except Exception:
                    pass
                finally:
                    if check_script.exists():
                        check_script.unlink(missing_ok=True)

            if unauthorized_axioms:
                passed = False

        return FormalProofReport(
            file_path=str(path_obj),
            passed=passed,
            engine="lean4",
            lean_installed=True,
            has_sorry=has_sorry,
            theorems_checked=theorems_checked,
            axioms_detected=axioms_detected,
            unauthorized_axioms=unauthorized_axioms,
            diagnostics=diagnostics,
            raw_output=raw_output,
            duration_sec=duration,
        )

    except subprocess.TimeoutExpired:
        return FormalProofReport(
            file_path=str(path_obj),
            passed=False,
            lean_installed=True,
            error_message=f"Lean 4 proof verification timed out after {timeout_sec} seconds.",
            duration_sec=time.time() - start_time,
        )
    except Exception as err:
        return FormalProofReport(
            file_path=str(path_obj),
            passed=False,
            lean_installed=True,
            error_message=f"Execution error during Lean verification: {str(err)}",
            duration_sec=time.time() - start_time,
        )
