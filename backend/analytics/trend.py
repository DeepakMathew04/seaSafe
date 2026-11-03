"""
Trend Estimation & Statistical Significance Engine (Developer 2).

Scope & Responsibilities:
1. Mann-Kendall Monotonic Trend Test:
   - Evaluates rank trend statistic S, test variance, and standard normal Z score.
   - Applies Hamed-Rao (1998) autocorrelation variance correction.
   - Integrates pymannkendall (original_test & seasonal_test) and native scipy implementations.
2. Theil-Sen Median Slope & 95% Confidence Interval:
   - Non-parametric median pairwise slope (scipy.stats.theilslopes).
   - 95% confidence interval [CI_lower, CI_upper] resistant to outliers.
3. Unit Normalization:
   - Converts raw month-step slopes into standard scientific reporting units:
     °C/decade (slope * 120) and °C/year (slope * 12).
4. OLS Baseline & Plain-English Interpretation.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
try:
    import pymannkendall as pmk
    HAS_PYMK = True
except ImportError:
    HAS_PYMK = False

from backend.analytics.autocorrelation import (
    compute_autocorrelation_function,
    calculate_hamed_rao_variance_correction
)

logger = logging.getLogger("SeaSafe.Trend")


def compute_ols_baseline(y: np.ndarray, x: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Ordinary Least Squares (OLS) linear regression baseline.

    Returns:
        {
            'slope_per_month': float,
            'slope_per_decade': float,
            'slope_per_year': float,
            'intercept': float,
            'r_squared': float,
            'p_value': float
        }
    """
    y = np.asarray(y, dtype=float)
    valid_idx = np.isfinite(y)
    y_clean = y[valid_idx]
    if x is None:
        x_clean = np.arange(len(y))[valid_idx]
    else:
        x_clean = np.asarray(x, dtype=float)[valid_idx]

    if len(y_clean) < 3:
        return {
            "slope_per_month": 0.0,
            "slope_per_decade": 0.0,
            "slope_per_year": 0.0,
            "intercept": float(np.mean(y_clean)) if len(y_clean) > 0 else 0.0,
            "r_squared": 0.0,
            "p_value": 1.0
        }

    res = stats.linregress(x_clean, y_clean)
    slope_month = float(res.slope)

    return {
        "slope_per_month": round(slope_month, 6),
        "slope_per_decade": round(slope_month * 120.0, 4),
        "slope_per_year": round(slope_month * 12.0, 5),
        "intercept": round(float(res.intercept), 4),
        "r_squared": round(float(res.rvalue ** 2), 4),
        "p_value": round(float(res.pvalue), 6)
    }


def calculate_theil_sen_trend(
    series: np.ndarray,
    alpha: float = 0.95
) -> Dict[str, Any]:
    """
    Compute Theil-Sen robust non-parametric slope and exact rank-based confidence interval.

    Formula:
        beta_Sen = median( (x_j - x_i) / (j - i) )  for all i < j

    Scientific Units:
        - °C / month: Raw pairwise median
        - °C / year: slope * 12
        - °C / decade: slope * 120
    """
    data = np.asarray(series, dtype=float)
    valid_idx = np.isfinite(data)
    y = data[valid_idx]
    x = np.arange(len(data))[valid_idx]

    if len(y) < 3:
        return {
            "slope_per_month": 0.0,
            "slope_deg_per_decade": 0.0,
            "slope_deg_per_year": 0.0,
            "ci_95": [0.0, 0.0],
            "ci_95_deg_per_decade": [0.0, 0.0],
            "intercept": float(np.mean(y)) if len(y) > 0 else 0.0
        }

    # Use scipy.stats.theilslopes
    res = stats.theilslopes(y, x, alpha=alpha)
    slope_m = float(res.slope)
    low_m = float(res.low_slope)
    high_m = float(res.high_slope)
    intercept = float(res.intercept)

    return {
        "slope_per_month": round(slope_m, 6),
        "slope_deg_per_decade": round(slope_m * 120.0, 4),
        "slope_deg_per_year": round(slope_m * 12.0, 5),
        "ci_95_deg_per_decade": [round(low_m * 120.0, 4), round(high_m * 120.0, 4)],
        "ci_95_deg_per_year": [round(low_m * 12.0, 5), round(high_m * 12.0, 5)],
        "intercept": round(intercept, 4)
    }


def run_mann_kendall_test(
    series: np.ndarray,
    adjust_autocorrelation: bool = True
) -> Dict[str, Any]:
    """
    Execute Mann-Kendall monotonic trend test with Hamed & Rao (1998) autocorrelation adjustment.

    Returns:
        {
            'tau': Kendall tau rank correlation,
            'p_value': Two-tailed p-value,
            'statistically_significant': bool (p < 0.05),
            'trend_direction': 'increasing', 'decreasing', or 'no_trend',
            's_statistic': int,
            'z_score': float,
            'autocorrelation_adjusted': bool,
            'variance_correction_eta': float
        }
    """
    data = np.asarray(series, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)

    if n < 4:
        return {
            "tau": 0.0,
            "p_value": 1.0,
            "statistically_significant": False,
            "trend_direction": "no_trend",
            "s_statistic": 0,
            "z_score": 0.0,
            "autocorrelation_adjusted": adjust_autocorrelation,
            "variance_correction_eta": 1.0
        }

    # Compute Mann-Kendall S statistic
    # S = sum_{i=1}^{n-1} sum_{j=i+1}^n sign(x_j - x_i)
    # diffs[i, j] = data[j] - data[i] for j > i (upper triangle)
    diffs = data[None, :] - data[:, None]
    s = int(np.sum(np.triu(np.sign(diffs), k=1)))

    # Compute ties
    unique_vals, tie_counts = np.unique(data, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))

    # Standard asymptotic variance V(S)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # Apply Hamed & Rao (1998) adjustment if requested
    eta = 1.0
    if adjust_autocorrelation:
        adj_var_s, eta, _ = calculate_hamed_rao_variance_correction(data, var_s)
        var_s = adj_var_s

    # Compute standard normal Z score
    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    # Kendall's Tau: tau = 2 * S / (n * (n - 1))
    tau = 2.0 * s / (n * (n - 1.0))

    trend_dir = "increasing" if (p_value < 0.05 and s > 0) else "decreasing" if (p_value < 0.05 and s < 0) else "no_trend"

    return {
        "tau": round(float(tau), 4),
        "p_value": round(float(p_value), 6),
        "statistically_significant": bool(p_value < 0.05),
        "trend_direction": trend_dir,
        "s_statistic": int(s),
        "z_score": round(float(z), 4),
        "autocorrelation_adjusted": adjust_autocorrelation,
        "variance_correction_eta": round(float(eta), 4)
    }


def run_seasonal_mann_kendall_test(series: np.ndarray, period: int = 12) -> Dict[str, Any]:
    """
    Run Seasonal Mann-Kendall test across 12 monthly seasons using pymannkendall or native algorithm.
    """
    data = np.asarray(series, dtype=float)
    if HAS_PYMK and len(data[np.isfinite(data)]) >= period * 2:
        try:
            res = pmk.seasonal_test(data, period=period)
            return {
                "trend": res.trend,
                "p_value": round(float(res.p), 6),
                "z_score": round(float(res.z), 4),
                "tau": round(float(res.Tau), 4),
                "slope_deg_per_decade": round(float(res.slope) * 120.0, 4),
                "statistically_significant": bool(res.h)
            }
        except Exception as e:
            logger.warning(f"pymannkendall seasonal_test error: {e}. Falling back to standard MK.")

    # Fallback to standard autocorrelation-adjusted Mann-Kendall
    mk = run_mann_kendall_test(data, adjust_autocorrelation=True)
    ts = calculate_theil_sen_trend(data)
    return {
        "trend": mk["trend_direction"],
        "p_value": mk["p_value"],
        "z_score": mk["z_score"],
        "tau": mk["tau"],
        "slope_deg_per_decade": ts["slope_deg_per_decade"],
        "statistically_significant": mk["statistically_significant"]
    }


def calculate_complete_trend_analysis(
    df: pd.DataFrame,
    value_col: str = "anomaly_celsius",
    time_col: str = "date"
) -> Dict[str, Any]:
    """
    Master trend analysis pipeline delivering clean, peer-reviewed statistical dictionaries.

    Returns:
        {
            'trend_deg_per_decade': float,
            'trend_deg_per_year': float,
            'ci_95': [lower, upper],
            'p_value': float,
            'statistically_significant': bool,
            'method': str,
            'ols_baseline_deg_per_decade': float,
            'ols_r_squared': float,
            'autocorrelation_acf': dict,
            'interpretation': str
        }
    """
    clean_df = df.dropna(subset=[value_col]).sort_values(time_col)
    values = clean_df[value_col].values

    # 1. Theil-Sen Median Slope & 95% Confidence Interval
    theil_sen = calculate_theil_sen_trend(values)

    # 2. Mann-Kendall with Hamed-Rao Variance Correction
    mk = run_mann_kendall_test(values, adjust_autocorrelation=True)

    # 3. Autocorrelation Function (ACF) diagnostic
    acf = compute_autocorrelation_function(values, max_lags=24)

    # 4. OLS Baseline
    ols = compute_ols_baseline(values)

    # 5. Scientific Plain-English Interpretation
    slope_dec = theil_sen["slope_deg_per_decade"]
    ci_low, ci_high = theil_sen["ci_95_deg_per_decade"]
    p_val = mk["p_value"]

    if mk["statistically_significant"]:
        dir_word = "warming" if slope_dec > 0 else "cooling"
        interpretation = (
            f"Statistically significant {dir_word} observed over {clean_df['year'].min()}–{clean_df['year'].max()} "
            f"({slope_dec:+.2f} °C/decade [95% CI: {ci_low:.2f} to {ci_high:.2f}], p = {p_val:.4f})."
        )
    else:
        interpretation = (
            f"No statistically significant monotonic trend detected over {clean_df['year'].min()}–{clean_df['year'].max()} "
            f"({slope_dec:+.2f} °C/decade [95% CI: {ci_low:.2f} to {ci_high:.2f}], p = {p_val:.4f} >= 0.05)."
        )

    return {
        "trend_deg_per_decade": slope_dec,
        "trend_deg_per_year": theil_sen["slope_deg_per_year"],
        "ci_95": [ci_low, ci_high],
        "p_value": p_val,
        "statistically_significant": mk["statistically_significant"],
        "method": "Sen's slope + Mann-Kendall (Hamed-Rao autocorrelation-adjusted)",
        "ols_baseline_deg_per_decade": ols["slope_per_decade"],
        "ols_r_squared": ols["r_squared"],
        "autocorrelation_acf": {
            "has_significant_autocorrelation": acf["has_significant_autocorrelation"],
            "variance_correction_eta": mk["variance_correction_eta"],
            "significant_lags": acf["significant_lags"]
        },
        "interpretation": interpretation
    }
