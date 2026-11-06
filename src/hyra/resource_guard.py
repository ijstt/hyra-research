import os
import psutil
import signal
import subprocess
import time
from typing import Any, Dict, Optional, Tuple


def get_system_metrics() -> Dict[str, Any]:
    """Retrieve host system hardware telemetry (RAM, CPU, GPU VRAM)."""
    vm = psutil.virtual_memory()
    metrics: Dict[str, Any] = {
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_percent": vm.percent,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "gpu_available": False,
        "gpu_vram_free_mb": None,
        "gpu_vram_total_mb": None,
        "gpu_name": None,
    }

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                metrics["gpu_available"] = True
                metrics["gpu_name"] = parts[0]
                metrics["gpu_vram_free_mb"] = int(parts[1])
                metrics["gpu_vram_total_mb"] = int(parts[2])
    except Exception:
        pass

    return metrics


def run_sandboxed_command(
    cmd: str,
    cwd: str,
    timeout_sec: int = 1800,
    env_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float]:
    """Execute a sandboxed shell command with timeout and resource tracking.
    
    Args:
        cmd: Shell command string to execute.
        cwd: Working directory.
        timeout_sec: Maximum permitted execution time before SIGKILL.
        env_overrides: Optional environment variables dictionary.
        
    Returns:
        Tuple of (returncode, stdout, stderr, elapsed_seconds).
    """
    start_time = time.time()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
    }
    # On POSIX systems, spawn into a new process group so child processes can be terminated together
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        ret_code = proc.returncode
    except subprocess.TimeoutExpired:
        # Terminate entire process tree (process group)
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            proc.kill()

        try:
            stdout, stderr = proc.communicate(timeout=5)
        except Exception:
            stdout, stderr = "", ""
        ret_code = -1
        stderr += f"\n[RESOURCE_GUARD]: Process group terminated forcefully due to timeout ({timeout_sec}s)."

    elapsed = round(time.time() - start_time, 2)
    return ret_code, stdout, stderr, elapsed
