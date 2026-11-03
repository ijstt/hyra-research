import numpy as np

def simulate_lif(current: np.ndarray, dt: float = 0.001, v_rest: float = -70.0, v_thresh: float = -50.0, tau_m: float = 0.02) -> tuple:
    n_steps = len(current)
    v = np.zeros(n_steps)
    v[0] = v_rest
    spikes = np.zeros(n_steps, dtype=bool)

    for t in range(1, n_steps):
        dv = (-(v[t-1] - v_rest) + current[t]) * (dt / tau_m)
        v[t] = v[t-1] + dv
        if v[t] >= v_thresh:
            spikes[t] = True
            v[t] = v_rest

    return v, spikes

if __name__ == "__main__":
    np.random.seed(42)
    current = 25.0 + np.random.normal(0, 5.0, 1000)
    v, spikes = simulate_lif(current)
    print(f"LIF Baseline Spikes: {spikes.sum()} / 1000 steps")
