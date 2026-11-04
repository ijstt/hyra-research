"""Tests for the hyra.leakage_detector module and check-leakage CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hyra.leakage_detector import (
    FileLeakageReport,
    LeakageFinding,
    LeakageReport,
    analyze_code,
    analyze_file,
    analyze_path,
)


def test_clean_ml_pipeline_passes():
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
"""
    report = analyze_code(code)
    assert report.passed
    assert len(report.findings) == 0


def test_pre_split_transformation_leakage():
    code = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
scaler = StandardScaler()
# Data leakage: scaler fitted on full dataset before train_test_split
df_scaled = scaler.fit_transform(df)

X = df_scaled.drop(columns=["target"])
y = df_scaled["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
    report = analyze_code(code)
    assert not report.passed
    rule_ids = [f.rule_id for f in report.findings]
    assert "HYRA-LEAK-002" in rule_ids


def test_fit_on_test_set_leakage():
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
# Data leakage: fitting on test set
scaler.fit(X_test)
"""
    report = analyze_code(code)
    assert not report.passed
    rule_ids = [f.rule_id for f in report.findings]
    assert "HYRA-LEAK-001" in rule_ids


def test_lookahead_negative_shift():
    code = """
import pandas as pd

df = pd.read_csv("prices.csv")
# Lookahead leak: future sample shifted backwards into feature
df["future_leak"] = df["close"].shift(-1)
df["future_leak_kw"] = df["close"].shift(periods=-2)
"""
    report = analyze_code(code)
    assert not report.passed
    shifts = [f for f in report.findings if f.rule_id == "HYRA-LOOKAHEAD-001"]
    assert len(shifts) == 2


def test_lookahead_centered_rolling():
    code = """
import pandas as pd

df = pd.read_csv("timeseries.csv")
# Lookahead leak: center=True incorporates forward-looking timestamps
df["rolling_mean"] = df["val"].rolling(window=5, center=True).mean()
"""
    report = analyze_code(code)
    assert not report.passed
    assert any(f.rule_id == "HYRA-LOOKAHEAD-002" for f in report.findings)


def test_target_in_feature_list():
    code = """
feature_columns = ["volume", "volatility", "target", "rsi"]
"""
    report = analyze_code(code)
    assert not report.passed
    assert any(f.rule_id == "HYRA-LEAK-003" for f in report.findings)


def test_raw_df_in_split_without_dropping_target():
    code = """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")
y = df["target"]
# Data leakage: passing df directly while y is target
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
"""
    report = analyze_code(code)
    assert not report.passed
    assert any(f.rule_id == "HYRA-LEAK-004" for f in report.findings)


def test_model_fit_with_target_in_dataframe():
    code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset.csv")
y = df["target"]
model = RandomForestClassifier()
# df was never dropped of target
model.fit(df, y)
"""
    report = analyze_code(code)
    assert not report.passed
    assert any(f.rule_id == "HYRA-LEAK-005" for f in report.findings)


def test_syntax_error_handling():
    code = "def broken_func(:\n    pass"
    report = analyze_code(code, filename="broken.py")
    assert not report.passed
    assert any(f.rule_id == "HYRA-SYNTAX-001" for f in report.findings)


def test_analyze_file_and_path(tmp_path: Path):
    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        "from sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y)\n",
        encoding="utf-8",
    )
    leak_file = tmp_path / "leaky.py"
    leak_file.write_text(
        "import pandas as pd\ndf = pd.read_csv('x.csv')\nx = df.shift(-1)\n",
        encoding="utf-8",
    )

    file_report = analyze_file(clean_file)
    assert file_report.passed

    dir_report = analyze_path(tmp_path)
    assert dir_report.total_files == 2
    assert not dir_report.passed
    assert dir_report.critical_count >= 1

    # Format terminal output validation
    term_str = dir_report.format_terminal()
    assert "HYRA ANTI-LEAKAGE STATIC CODE AUDIT REPORT" in term_str
    assert "FAILED" in term_str

    # Dictionary validation
    d = dir_report.to_dict()
    assert d["total_files"] == 2
    assert d["passed"] is False


def test_cli_check_leakage(tmp_path: Path):
    clean_script = tmp_path / "clean_model.py"
    clean_script.write_text(
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        encoding="utf-8",
    )

    # 1. Test clean file passes with exit code 0
    res = subprocess.run(
        [sys.executable, "-m", "hyra.cli", "check-leakage", str(clean_script)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Zero critical data leakage detected" in res.stdout

    # 2. Test leaky file fails with exit code 1
    leaky_script = tmp_path / "leaky_model.py"
    leaky_script.write_text(
        "import pandas as pd\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "df = pd.read_csv('data.csv')\n"
        "scaler = StandardScaler()\n"
        "scaler.fit(df)\n",
        encoding="utf-8",
    )
    res_leak = subprocess.run(
        [sys.executable, "-m", "hyra.cli", "check-leakage", str(leaky_script)],
        capture_output=True,
        text=True,
    )
    assert res_leak.returncode == 1
    assert "HYRA-LEAK-002" in res_leak.stdout

    # 3. Test --json output
    res_json = subprocess.run(
        [sys.executable, "-m", "hyra.cli", "check-leakage", str(leaky_script), "--json"],
        capture_output=True,
        text=True,
    )
    data = json.loads(res_json.stdout)
    assert data["passed"] is False
    assert data["critical_count"] >= 1
