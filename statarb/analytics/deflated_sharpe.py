import numpy as np
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015328


def _sr_estimator_std(returns: np.ndarray, annualization: int = 252) -> float:
    T = len(returns)
    if T < 4:
        return np.inf
    sr = returns.mean() / returns.std(ddof=1) * np.sqrt(annualization)
    skew = float(stats.skew(returns))
    # excess kurtosis
    kurt = float(stats.kurtosis(returns, bias=False))
    # annualized SR, convert back to daily-unit before computing se
    sr_daily = returns.mean() / returns.std(ddof=1)
    variance = (1 + 0.5 * sr_daily**2 - skew * sr_daily + (kurt / 4) * sr_daily**2) / T
    return float(np.sqrt(max(variance, 0)) * np.sqrt(annualization))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    if n_trials <= 1:
        return 0.0
    em = _EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return sr_std * ((1.0 - em) * z1 + em * z2)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    annualization: int = 252,
) -> float:
    T = len(returns)
    sr_hat = returns.mean() / returns.std(ddof=1) * np.sqrt(annualization)
    sr_std = _sr_estimator_std(returns, annualization)
    if sr_std == 0:
        return float(sr_hat > sr_benchmark)
    z = (sr_hat - sr_benchmark) / sr_std
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    returns: np.ndarray,
    n_trials: int,
    annualization: int = 252,
) -> dict:
    sr_std = _sr_estimator_std(returns, annualization)
    sr_max = expected_max_sharpe(n_trials, sr_std)

    if sr_std == 0:
        dsr_z = 0.0
    else:
        dsr_z = (observed_sr - sr_max) / sr_std

    prob_positive = float(stats.norm.cdf(dsr_z))

    return {
        "observed_sr": observed_sr,
        "expected_max_sr_over_trials": sr_max,
        "sr_estimator_std": sr_std,
        "dsr_z_score": dsr_z,
        "prob_true_sr_positive": prob_positive,
        "n_trials": n_trials,
    }
