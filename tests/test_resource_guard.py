from hyra.resource_guard import get_system_metrics, run_sandboxed_command


def test_system_metrics():
    metrics = get_system_metrics()
    assert "ram_total_gb" in metrics
    assert "ram_available_gb" in metrics
    assert "gpu_available" in metrics
    assert metrics["ram_total_gb"] > 0


def test_sandboxed_command_success(tmp_path):
    ret, stdout, stderr, elapsed = run_sandboxed_command("echo 'Hello Hyra'", cwd=str(tmp_path))
    assert ret == 0
    assert "Hello Hyra" in stdout
    assert elapsed >= 0.0


def test_sandboxed_command_timeout(tmp_path):
    ret, stdout, stderr, elapsed = run_sandboxed_command("sleep 5", cwd=str(tmp_path), timeout_sec=1)
    assert ret == -1
    assert "timeout" in stderr.lower()
