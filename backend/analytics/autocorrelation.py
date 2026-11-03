"""
Autocorrelation & Serial Dependence Analysis (Developer 2).

Scope & Responsibilities:
1. Autocorrelation Function (ACF): Tests time series serial correlation across monthly lags (1 to 24).
2. Hamed & Rao (1998) Variance Correction: Corrects Mann-Kendall variance inflation caused by
   significant lag autocorrelation, ensuring statistical significance is neither inflated nor biased.
"""

import logging
from typing import Tuple, Dict, Any, List
import numpy as np

logger = logging.getLogger("SeaSafe.Autocorrelation")


def compute_autocorrelation_function(series: np.ndarray, max_lags: int = 24) -> Dict[str, Any]:
    """
    Compute sample Autocorrelation Function (ACF) and 95% Bartlett significance bounds.

    Parameters:
        series: 1D numerical array of observations or anomalies.
        max_lags: Maximum lag to evaluate (default 24 months = 2 annual cycles).

    Returns:
        Dict with:
            - 'lags': List of lag integers [1, 2, ..., max_lags]
            - 'acf_values': List of autocorrelation coefficients r_k
            - 'confidence_threshold_95': Two-tailed significance boundary (+/- 1.96 / sqrt(N))
            - 'has_significant_autocorrelation': Boolean flag indicating if lag-1 or lag-12 exceeds bounds
            - 'significant_lags': List of lags exceeding 95% bounds
    """
    data = np.asarray(series, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)

    if n < 4:
        return {
            "lags": [],
            "acf_values": [],
            "confidence_threshold_95": 1.0,
            "has_significant_autocorrelation": False,
            "significant_lags": []
        }

    mean_val = np.mean(data)
    variance = np.var(data)
    if variance < 1e-9:
        return {
            "lags": list(range(1, max_lags + 1)),
            "acf_values": [0.0] * max_lags,
            "confidence_threshold_95": 1.96 / np.sqrt(n),
            "has_significant_autocorrelation": False,
            "significant_lags": []
        }

    max_l = min(max_lags, n - 2)
    acf_vals = []
    lags = list(range(1, max_l + 1))
    threshold_95 = 1.96 / np.sqrt(n)

    for k in lags:
        r_k = np.sum((data[:-k] - mean_val) * (data[k:] - mean_val)) / (n * variance)
        acf_vals.append(float(r_k))

    significant_lags = [lag for lag, r in zip(lags, acf_vals) if abs(r) > threshold_95]
    has_sig = len(significant_lags) > 0

    return {
        "lags": lags,
        "acf_values": [round(v, 4) for v in acf_vals],
        "confidence_threshold_95": round(float(threshold_95), 4),
        "has_significant_autocorrelation": has_sig,
        "significant_lags": significant_lags
    }


def calculate_hamed_rao_variance_correction(
    series: np.ndarray,
    original_var_s: float
) -> Tuple[float, float, List[int]]:
    """
    Calculate the Hamed & Rao (1998) variance correction factor for Mann-Kendall test.

    Formula:
        V*(S) = V(S) * eta
        eta = 1 + (2 / (n * (n - 1) * (n - 2))) * sum_{i=1}^{n-1} (n - i)(n - i - 1)(n - i - 2) * rho(i)

    Where:
        rho(i) is the empirical rank autocorrelation at lag i.
        Only statistically significant rank autocorrelations are included in the sum.

    Returns:
        (adjusted_var_s, correction_factor_eta, significant_rank_lags)
    """
    data = np.asarray(series, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)

    if n < 4:
        return original_var_s, 1.0, []

    # Compute ranks
    temp = data.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(n, dtype=float)

    rank_mean = (n - 1.0) / 2.0
    rank_var = np.var(ranks)

    if rank_var < 1e-9:
        return original_var_s, 1.0, []

    threshold = 1.96 / np.sqrt(n)
    sum_terms = 0.0
    significant_rank_lags = []

    # Evaluate rank autocorrelations up to n-1 (practical limit min(36, n-2))
    max_k = min(36, n - 2)
    for k in range(1, max_k + 1):
        r_k = np.sum((ranks[:-k] - rank_mean) * (ranks[k:] - rank_mean)) / (n * rank_var)
        if abs(r_k) > threshold:
            significant_rank_lags.append(k)
            weight = (n - k) * (n - k - 1) * (n - k - 2)
            sum_terms += weight * r_k

    eta = 1.0 + (2.0 / (n * (n - 1.0) * (n - 2.0))) * sum_terms

    # Variance scaling factor must be strictly positive
    if eta <= 0.0 or not np.isfinite(eta):
        eta = 1.0

    adjusted_var_s = original_var_s * eta
    return float(adjusted_var_s), float(round(eta, 4)), significant_rank_lags
