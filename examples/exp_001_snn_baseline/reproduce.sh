#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[HYRA-REPRODUCE]: Running EXP-001 Baseline and Proposed Model benchmark..."
python3 -c "
import numpy as np
from src.baseline import simulate_lif
from src.model import simulate_adaptive_lif

np.random.seed(42)
signal = 25.0 + np.random.normal(0, 5.0, 5000)
_, sp_base = simulate_lif(signal)
_, _, sp_adapt = simulate_adaptive_lif(signal)

print(f'Baseline Spikes Count: {sp_base.sum()}')
print(f'Adaptive Spikes Count: {sp_adapt.sum()}')
print('Relative burst reduction: {:.2f}%'.format((1 - sp_adapt.sum() / sp_base.sum()) * 100))
"
echo "[HYRA-REPRODUCE]: Execution completed successfully with exit code 0."
