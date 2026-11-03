import numpy as np

def simulate_adaptive_lif(
    current: np.ndarray,
    dt: float = 0.001,
    v_rest: float = -70.0,
    theta_0: float = -50.0,
    tau_m: float = 0.02,
    tau_theta: float = 0.05,
    delta_theta: float = 5.0,
) -> tuple:
    n_steps = len(current)
    v = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    v[0] = v_rest
    theta[0] = theta_0
    spikes = np.zeros(n_steps, dtype=bool)

    for t in range(1, n_steps):
        dv = (-(v[t-1] - v_rest) + current[t]) * (dt / tau_m)
        d_theta = (-(theta[t-1] - theta_0)) * (dt / tau_theta)

        v[t] = v[t-1] + dv
        theta[t] = theta[t-1] + d_theta

        if v[t] >= theta[t]:
            spikes[t] = True
            v[t] = v_rest
            theta[t] += delta_theta

    return v, theta, spikes

if __name__ == "__main__":
    np.random.seed(42)
    current = 25.0 + np.random.normal(0, 5.0, 1000)
    v, theta, spikes = simulate_adaptive_lif(current)
    print(f"Adaptive LIF Spikes: {spikes.sum()} / 1000 steps")
