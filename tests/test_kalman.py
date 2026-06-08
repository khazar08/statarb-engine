"""
Kalman filter tests.

1. On a synthetic series with a known constant beta, the filter should
   converge to that beta within a reasonable number of observations.

2. A time-varying beta should be tracked: after a step change the filter
   should converge to the new value.

3. batch_filter must return arrays of the same length as the input.
"""

import numpy as np
import pytest

from statarb.strategy.kalman import KalmanHedgeRatio, batch_filter


def _synthetic_series(beta: float, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    log_x = np.cumsum(rng.normal(0, 0.01, n))
    noise = rng.normal(0, 0.005, n)
    log_y = 0.2 + beta * log_x + noise  # alpha=0.2, known beta
    return log_x, log_y


def test_kalman_converges_to_known_beta():
    true_beta = 1.5
    log_x, log_y = _synthetic_series(true_beta)

    kf = KalmanHedgeRatio(delta=1e-5, obs_noise=1e-3)
    alpha_est = beta_est = None
    for lx, ly in zip(log_x, log_y):
        alpha_est, beta_est = kf.update(lx, ly)

    assert abs(beta_est - true_beta) < 0.05, (
        f"Filter did not converge: estimated beta={beta_est:.4f}, true={true_beta}"
    )
    assert abs(alpha_est - 0.2) < 0.1, (
        f"Filter did not converge: estimated alpha={alpha_est:.4f}, true=0.2"
    )


def test_kalman_tracks_step_change():
    """After a step change in beta, the filter should re-converge.

    Using step=0.05 for log_x so beta is well-identified (small x keeps beta
    shrunk toward zero and makes convergence very slow).
    """
    rng = np.random.default_rng(1)
    n = 400
    log_x = np.cumsum(rng.normal(0, 0.05, n))  # larger steps → β more identifiable
    noise = rng.normal(0, 0.01, n)

    betas = np.where(np.arange(n) < n // 2, 1.0, 2.0)
    log_y = betas * log_x + noise

    kf = KalmanHedgeRatio(delta=5e-4, obs_noise=1e-2)
    for lx, ly in zip(log_x[:200], log_y[:200]):
        kf.update(lx, ly)
    beta_mid = kf.beta
    assert abs(beta_mid - 1.0) < 0.25, f"Pre-change beta estimate off: {beta_mid:.3f}"

    for lx, ly in zip(log_x[200:], log_y[200:]):
        kf.update(lx, ly)
    beta_end = kf.beta
    # must have moved toward 2.0 from wherever it was mid-way
    assert beta_end > beta_mid, "Filter should increase beta after step-up"
    assert abs(beta_end - 2.0) < 0.30, f"Post-change beta estimate off: {beta_end:.3f}"


def test_batch_filter_length():
    log_x, log_y = _synthetic_series(0.8, n=200)
    alphas, betas = batch_filter(log_x, log_y)
    assert len(alphas) == 200
    assert len(betas) == 200


def test_kalman_reset():
    kf = KalmanHedgeRatio()
    for _ in range(50):
        kf.update(1.0, 1.5)
    kf.reset()
    assert np.allclose(kf.theta, 0.0)
    assert np.allclose(kf.P, np.eye(2))
    assert kf._n_updates == 0


def test_invalid_delta_raises():
    with pytest.raises(ValueError):
        KalmanHedgeRatio(delta=1.5)
    with pytest.raises(ValueError):
        KalmanHedgeRatio(delta=0.0)
