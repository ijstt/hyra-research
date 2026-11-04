"""Autonomous AST Data Leakage Analyzer for ML/AI Research Pipelines.

Performs static analysis of Python source code to detect common data leakage
antipatterns before model training or evaluation:
1. Pre-split transformations: fitting scalers/encoders/pipelines before data splitting
   or fitting them on test/validation sets.
2. Lookahead bias in time series: negative shift calls or centered rolling/expanding windows.
3. Target leakage: target column retained in feature matrix or included in feature list.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class LeakageFinding:
    category: str
    severity: str
    line: int
    col: int
    message: str
    recommendation: str
    code_snippet: Optional[str] = None
    rule_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileLeakageReport:
    file_path: str
    findings: List[LeakageFinding] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "passed": self.passed,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class LeakageReport:
    total_files: int = 0
    total_findings: int = 0
    critical_count: int = 0
    warning_count: int = 0
    passed: bool = True
    files: List[FileLeakageReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_findings": self.total_findings,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "passed": self.passed,
            "files": [f.to_dict() for f in self.files],
        }

    def format_terminal(self) -> str:
        lines: List[str] = []
        border = "=" * 78
        lines.append(border)
        lines.append("HYRA ANTI-LEAKAGE STATIC CODE AUDIT REPORT")
        lines.append(border)
        lines.append(
            f"Files analyzed: {self.total_files} | Total findings: {self.total_findings} | "
            f"Critical: {self.critical_count} | Warnings: {self.warning_count}"
        )
        verdict = "PASSED (Zero critical data leakage detected)" if self.passed else "FAILED (Data leakage vulnerabilities detected)"
        lines.append(f"Verdict: {verdict}")
        lines.append(border)

        for file_report in self.files:
            if not file_report.findings:
                continue
            lines.append(f"\nFile: {file_report.file_path}")
            lines.append("-" * 78)
            for f in file_report.findings:
                lines.append(f"[{f.severity}] Line {f.line}:{f.col} - Rule: {f.rule_id} ({f.category})")
                lines.append(f"  Message: {f.message}")
                if f.code_snippet:
                    lines.append(f"  Code: {f.code_snippet.strip()}")
                lines.append(f"  Remediation: {f.recommendation}")
                lines.append("")

        if not self.total_findings:
            lines.append("\nAll inspected files passed anti-leakage static validation.")

        lines.append(border)
        return "\n".join(lines)


KNOWN_TRANSFORMER_CLASSES: Set[str] = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "Normalizer",
    "QuantileTransformer",
    "PowerTransformer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "LabelEncoder",
    "SimpleImputer",
    "IterativeImputer",
    "KNNImputer",
    "Pipeline",
    "ColumnTransformer",
    "PCA",
    "FastICA",
    "TruncatedSVD",
}

KNOWN_SPLIT_FUNCTIONS: Set[str] = {
    "train_test_split",
    "TimeSeriesSplit",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "PredefinedSplit",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
}

TEST_KEYWORDS: Set[str] = {
    "test",
    "val",
    "valid",
    "validation",
    "holdout",
    "eval",
    "evaluation",
}


def _get_call_name(node: ast.AST) -> Optional[str]:
    """Helper to extract simple function name or method name from Call node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _get_negative_number(node: ast.AST) -> Optional[float]:
    """Check if AST node evaluates to a negative number."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return -float(node.operand.value)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if node.value < 0:
            return float(node.value)
    return None


def _extract_constant_str(node: ast.AST) -> Optional[str]:
    """Extract string constant from AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class LeakageASTVisitor(ast.NodeVisitor):
    """AST visitor traversing a Python AST to spot data leakage antipatterns."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.findings: List[LeakageFinding] = []

        # State tracking across AST nodes
        self.transformer_vars: Set[str] = set()
        self.split_events: List[int] = []  # Line numbers of split function calls
        self.split_dataset_vars: Set[str] = set()
        self.test_dataset_vars: Set[str] = set()
        self.raw_dataset_vars: Set[str] = {"df", "data", "dataset", "raw_data", "raw_df"}
        self.dfs_with_target: Set[str] = set()
        self.target_col_names: Set[str] = {"target", "label", "y", "close", "target_col"}
        self.target_var_names: Set[str] = {"y", "target", "labels", "y_train", "y_test", "y_val"}
        self.dropped_target_dfs: Set[str] = set()

    def _get_snippet(self, node: ast.AST) -> str:
        line_idx = getattr(node, "lineno", 1) - 1
        if 0 <= line_idx < len(self.source_lines):
            return self.source_lines[line_idx].strip()
        return ""

    def visit_Assign(self, node: ast.Assign) -> None:
        # 1. Identify transformer instantiations: e.g. scaler = StandardScaler()
        if isinstance(node.value, ast.Call):
            call_func = _get_call_name(node.value.func)
            if call_func and (call_func in KNOWN_TRANSFORMER_CLASSES or "scaler" in call_func.lower() or "encoder" in call_func.lower()):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.transformer_vars.add(target.id)

        # 2. Check variable names for transformers by convention
        for target in node.targets:
            if isinstance(target, ast.Name):
                name_lower = target.id.lower()
                if any(k in name_lower for k in ("scaler", "encoder", "imputer")) and not any(k in name_lower for k in ("scale", "scaled")):
                    self.transformer_vars.add(target.id)

        # 3. Identify target definitions: e.g. target = 'price' or target_col = 'label'
        for target in node.targets:
            if isinstance(target, ast.Name):
                name_lower = target.id.lower()
                if "target" in name_lower or name_lower == "label_col":
                    val_str = _extract_constant_str(node.value)
                    if val_str:
                        self.target_col_names.add(val_str)

        # 4. Identify target series extraction: e.g. y = df['target'] or y = df[target_col]
        if isinstance(node.value, ast.Subscript):
            slice_val = _extract_constant_str(node.value.slice)
            parent_df = _get_call_name(node.value.value)
            if slice_val and (slice_val in self.target_col_names or "target" in slice_val.lower() or "label" in slice_val.lower()):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.target_var_names.add(target.id)
                if parent_df:
                    self.dfs_with_target.add(parent_df)

        # 5. Identify dataframe alias inheriting target without drop: e.g. X = df
        if isinstance(node.value, ast.Name):
            src_name = node.value.id
            if src_name in self.dfs_with_target:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dfs_with_target.add(target.id)

        # 6. Identify dataframe dropping target: df_feat = df.drop(columns=['target'])
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "drop":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dropped_target_dfs.add(target.id)

        # 6. Identify feature list definitions containing target column:
        # e.g. features = ['open', 'high', 'low', 'target']
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for elt in node.value.elts:
                col_name = _extract_constant_str(elt)
                if col_name and (col_name in self.target_col_names or col_name.lower() in ("target", "label")):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and any(k in target.id.lower() for k in ("feat", "feature", "cols", "columns", "x_cols")):
                            self.findings.append(
                                LeakageFinding(
                                    category="TARGET_LEAKAGE",
                                    severity="CRITICAL",
                                    line=node.lineno,
                                    col=node.col_offset,
                                    message=f"Target column '{col_name}' detected inside feature list '{target.id}'.",
                                    recommendation=f"Exclude '{col_name}' from input features {target.id} before model training.",
                                    code_snippet=self._get_snippet(node),
                                    rule_id="HYRA-LEAK-003",
                                )
                            )

        # 7. Identify train/test split assignments:
        # e.g. X_train, X_test, y_train, y_test = train_test_split(...)
        # or train, test = ...
        if isinstance(node.targets[0], (ast.Tuple, ast.List)):
            target_names = [elt.id for elt in node.targets[0].elts if isinstance(elt, ast.Name)]
            for name in target_names:
                self.split_dataset_vars.add(name)
                name_lower = name.lower()
                if any(kw in name_lower for kw in TEST_KEYWORDS):
                    self.test_dataset_vars.add(name)

        # Continue traversing children
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _get_call_name(node.func)
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        snippet = self._get_snippet(node)

        # A. Split event detection
        if call_name in KNOWN_SPLIT_FUNCTIONS:
            self.split_events.append(line)

            # Check for direct target leakage into features in train_test_split(df, y)
            if call_name == "train_test_split" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Name):
                    is_target_leaked = (
                        (first_arg.id in self.dfs_with_target or first_arg.id in self.raw_dataset_vars)
                        and first_arg.id not in self.dropped_target_dfs
                        and len(node.args) > 1
                        and (first_arg.id in self.dfs_with_target or any(k in first_arg.id.lower() for k in ("df", "data", "dataset")))
                    )
                    if is_target_leaked:
                        self.findings.append(
                            LeakageFinding(
                                category="TARGET_LEAKAGE",
                                severity="CRITICAL",
                                line=line,
                                col=col,
                                message=(
                                    f"Raw DataFrame '{first_arg.id}' passed as feature matrix to train_test_split "
                                    "without dropping target column."
                                ),
                                recommendation=(
                                    f"Drop target column from '{first_arg.id}' before splitting (e.g., "
                                    f"X = {first_arg.id}.drop(columns=['target']))."
                                ),
                                code_snippet=snippet,
                                rule_id="HYRA-LEAK-004",
                            )
                        )

        # B. Transformer .fit() or .fit_transform() checks
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("fit", "fit_transform"):
            caller_node = node.func.value
            caller_name = _get_call_name(caller_node)
            is_transformer = False

            if caller_name and (
                caller_name in self.transformer_vars
                or any(k in caller_name.lower() for k in ("scaler", "encoder", "imputer", "pipe", "pipeline"))
            ):
                is_transformer = True

            # If transformer is detected
            if is_transformer and node.args:
                arg0 = node.args[0]
                arg0_name = _get_call_name(arg0) or ""
                arg0_lower = arg0_name.lower()

                # Case 1: Fitting on test/validation dataset
                is_test_arg = (
                    arg0_name in self.test_dataset_vars
                    or any(kw in arg0_lower for kw in TEST_KEYWORDS)
                )
                if is_test_arg:
                    self.findings.append(
                        LeakageFinding(
                            category="PRE_SPLIT_TRANSFORMATION",
                            severity="CRITICAL",
                            line=line,
                            col=col,
                            message=(
                                f"Transformer '{caller_name}.{node.func.attr}()' is fitted directly on test/validation "
                                f"dataset '{arg0_name}'."
                            ),
                            recommendation=(
                                f"Fit '{caller_name}' exclusively on training data (e.g., {caller_name}.fit(X_train)), "
                                f"then use {caller_name}.transform({arg0_name}) on test/validation partitions."
                            ),
                            code_snippet=snippet,
                            rule_id="HYRA-LEAK-001",
                        )
                    )

                # Case 2: Fitting on full/raw dataset before split
                has_split = len(self.split_events) > 0 and line > min(self.split_events)
                if not has_split and (arg0_name in self.raw_dataset_vars or "df" in arg0_lower or arg0_name == "X"):
                    self.findings.append(
                        LeakageFinding(
                            category="PRE_SPLIT_TRANSFORMATION",
                            severity="CRITICAL",
                            line=line,
                            col=col,
                            message=(
                                f"Transformer '{caller_name}.{node.func.attr}()' called on full dataset '{arg0_name}' "
                                "before data split."
                            ),
                            recommendation=(
                                "Split data into train and test sets BEFORE applying transformations, "
                                f"and fit '{caller_name}' only on the training partition."
                            ),
                            code_snippet=snippet,
                            rule_id="HYRA-LEAK-002",
                        )
                    )

        # C. Lookahead bias: negative shift
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            neg_shift = None
            # Positional argument: .shift(-1)
            if node.args:
                neg_shift = _get_negative_number(node.args[0])
            # Keyword argument: .shift(periods=-1)
            for kw in node.keywords:
                if kw.arg == "periods":
                    neg_shift = _get_negative_number(kw.value)

            if neg_shift is not None and neg_shift < 0:
                self.findings.append(
                    LeakageFinding(
                        category="LOOKAHEAD_BIAS",
                        severity="CRITICAL",
                        line=line,
                        col=col,
                        message=f"Negative shift detected (shift({int(neg_shift)})), introducing lookahead bias.",
                        recommendation=(
                            "In time-series feature engineering, only use past lags (positive shift values). "
                            "If building future targets, verify this series is never used as an input feature."
                        ),
                        code_snippet=snippet,
                        rule_id="HYRA-LOOKAHEAD-001",
                    )
                )

        # D. Lookahead bias: centered rolling or expanding windows
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("rolling", "expanding"):
            for kw in node.keywords:
                if kw.arg == "center":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self.findings.append(
                            LeakageFinding(
                                category="LOOKAHEAD_BIAS",
                                severity="CRITICAL",
                                line=line,
                                col=col,
                                message=(
                                    f"Centered window aggregation '{node.func.attr}(..., center=True)' incorporates "
                                    "future samples into current window."
                                ),
                                recommendation=(
                                    f"Set 'center=False' in '{node.func.attr}()' to enforce causal, backwards-only "
                                    "aggregations for time-series models."
                                ),
                                code_snippet=snippet,
                                rule_id="HYRA-LOOKAHEAD-002",
                            )
                        )

        # E. Model fitting with target variable inside feature set:
        # e.g. model.fit(df, y) where df contains target
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            caller_name = _get_call_name(node.func.value) or ""
            if not any(k in caller_name.lower() for k in ("scaler", "encoder", "imputer")):
                if len(node.args) >= 2:
                    feat_arg = node.args[0]
                    target_arg = node.args[1]
                    feat_name = _get_call_name(feat_arg) or ""
                    target_name = _get_call_name(target_arg) or ""

                    if (
                        feat_name in self.raw_dataset_vars
                        and feat_name not in self.dropped_target_dfs
                        and (target_name in self.target_var_names or "y" in target_name.lower())
                    ):
                        self.findings.append(
                            LeakageFinding(
                                category="TARGET_LEAKAGE",
                                severity="CRITICAL",
                                line=line,
                                col=col,
                                message=(
                                    f"Model '{caller_name}.fit()' receives raw DataFrame '{feat_name}' as features while "
                                    f"target '{target_name}' was extracted from it without dropping."
                                ),
                                recommendation=f"Drop target column from '{feat_name}' before fitting model.",
                                code_snippet=snippet,
                                rule_id="HYRA-LEAK-005",
                            )
                        )

        self.generic_visit(node)


def analyze_code(code: str, filename: str = "<string>") -> FileLeakageReport:
    """Analyze Python source code string for data leakage vulnerabilities."""
    source_lines = code.splitlines()
    report = FileLeakageReport(file_path=filename)

    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as err:
        report.findings.append(
            LeakageFinding(
                category="SYNTAX_ERROR",
                severity="CRITICAL",
                line=err.lineno or 1,
                col=err.offset or 0,
                message=f"Syntax error parsing Python file: {err.msg}",
                recommendation="Fix Python syntax error before running anti-leakage audit.",
                code_snippet=err.text.strip() if err.text else None,
                rule_id="HYRA-SYNTAX-001",
            )
        )
        report.passed = False
        return report

    visitor = LeakageASTVisitor(source_lines)
    visitor.visit(tree)
    report.findings = visitor.findings
    report.passed = not any(f.severity == "CRITICAL" for f in report.findings)
    return report


def analyze_file(file_path: Path) -> FileLeakageReport:
    """Analyze a single Python file for data leakage vulnerabilities."""
    path_obj = Path(file_path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(f"Source file not found: {path_obj}")

    code = path_obj.read_text(encoding="utf-8")
    return analyze_code(code, filename=str(path_obj))


def analyze_path(target_path: Path) -> LeakageReport:
    """Analyze a file or recursively analyze a directory for data leakage."""
    path_obj = Path(target_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Target path not found: {path_obj}")

    file_reports: List[FileLeakageReport] = []

    if path_obj.is_file():
        if path_obj.suffix == ".py":
            file_reports.append(analyze_file(path_obj))
    else:
        for root, _, files in os.walk(path_obj):
            for file_name in sorted(files):
                if file_name.endswith(".py"):
                    full_p = Path(root) / file_name
                    # Skip virtual environments and hidden dirs relative to target_path
                    rel_p = full_p.relative_to(path_obj)
                    if any(part.startswith(".") or part in ("venv", ".venv", "__pycache__") for part in rel_p.parts[:-1]):
                        continue
                    file_reports.append(analyze_file(full_p))

    total_findings = sum(len(r.findings) for r in file_reports)
    crit_count = sum(sum(1 for f in r.findings if f.severity == "CRITICAL") for r in file_reports)
    warn_count = sum(sum(1 for f in r.findings if f.severity == "WARNING") for r in file_reports)
    overall_passed = crit_count == 0

    return LeakageReport(
        total_files=len(file_reports),
        total_findings=total_findings,
        critical_count=crit_count,
        warning_count=warn_count,
        passed=overall_passed,
        files=file_reports,
    )
