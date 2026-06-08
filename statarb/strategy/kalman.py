import numpy as np


class KalmanHedgeRatio:
    """
    Tracks a time-varying hedge ratio (and intercept) in the model:
        log(P_y) = alpha_t + beta_t * log(P_x) + eps_obs

    State vector: theta = [alpha, beta]^T
    Transition:   theta_t = theta_{t-1} + w_t   (random walk)
    Observation:  y_t = H_t @ theta_t + v_t

    Process noise:    Q = delta/(1-delta) * I  (Var(w))
    Observation noise: R  (Var(v))

    Both Q and R are tunable via `delta` and `obs_noise`.
    """

    def __init__(self, delta: float = 1e-4, obs_noise: float = 1e-2):
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.Q = delta / (1.0 - delta) * np.eye(2)  # process noise cov
        self.R = float(obs_noise)                    # scalar observation noise var
        self.theta = np.zeros(2)                     # [alpha, beta]
        self.P = np.eye(2)                           # state error covariance
        self._n_updates = 0

    def update(self, log_x: float, log_y: float) -> tuple[float, float]:
        """
        Incorporate one new (log_x, log_y) observation.
        Returns updated (alpha, beta).
        """
        H = np.array([1.0, log_x])  # observation row vector

        # predict
        P_pred = self.P + self.Q

        # innovation
        v = log_y - H @ self.theta

        # innovation variance (scalar)
        S = H @ P_pred @ H + self.R

        # Kalman gain (2-vector)
        K = (P_pred @ H) / S

        # state and covariance update
        self.theta = self.theta + K * v
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        self._n_updates += 1
        return float(self.theta[0]), float(self.theta[1])

    def reset(self) -> None:
        self.theta = np.zeros(2)
        self.P = np.eye(2)
        self._n_updates = 0

    @property
    def alpha(self) -> float:
        return float(self.theta[0])

    @property
    def beta(self) -> float:
        return float(self.theta[1])

    @property
    def hedge_ratio(self) -> float:
        return self.beta


def batch_filter(log_x: np.ndarray, log_y: np.ndarray,
                 delta: float = 1e-4, obs_noise: float = 1e-2
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Kalman filter over full arrays and return alpha, beta series.
    Useful for in-sample pair selection diagnostics.
    """
    kf = KalmanHedgeRatio(delta=delta, obs_noise=obs_noise)
    alphas = np.empty(len(log_x))
    betas = np.empty(len(log_x))
    for i, (lx, ly) in enumerate(zip(log_x, log_y)):
        alphas[i], betas[i] = kf.update(lx, ly)
    return alphas, betas
