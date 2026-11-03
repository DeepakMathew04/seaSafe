"""
Climatology & SST Anomaly Engine (Developer 2).

Scope & Responsibilities:
1. Baseline Reference Period: Default standard 2003-01-01 to 2020-12-31 (18-year baseline).
2. Climatological Normals: 12 calendar monthly means:
      Baseline(m) = (1/N) * sum_{y=1}^N SST(y, m)
3. Monthly SST Anomalies:
      Anomaly(y, m) = SST(y, m) - Baseline(m)
4. Standardized Z-Score & Extreme Event Detection:
      Z(y, m) = Anomaly(y, m) / sigma_m
   Flags extreme marine warming (|Z| >= 2.0) and intense upwelling cooling departures.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("SeaSafe.Climatology")

DEFAULT_BASELINE_START_YEAR = 2003
DEFAULT_BASELINE_END_YEAR = 2020


def calculate_monthly_climatology(
    monthly_df: pd.DataFrame,
    baseline_start_year: int = DEFAULT_BASELINE_START_YEAR,
    baseline_end_year: int = DEFAULT_BASELINE_END_YEAR
) -> Dict[int, Dict[str, float]]:
    """
    Compute 12 calendar monthly climatological normals from fixed reference window.

    Parameters:
        monthly_df: DataFrame with ['year', 'month', 'sst_mean_celsius']
        baseline_start_year: Start year (inclusive, default 2003)
        baseline_end_year: End year (inclusive, default 2020)

    Returns:
        Dict mapping month (1-12) to:
            {
                'mean': float (baseline mean in °C),
                'std': float (baseline standard deviation in °C),
                'n_years': int (sample count)
            }
    """
    baseline_mask = (
        (monthly_df["year"] >= baseline_start_year) &
        (monthly_df["year"] <= baseline_end_year) &
        monthly_df["sst_mean_celsius"].notna()
    )
    baseline_data = monthly_df[baseline_mask]

    if len(baseline_data) == 0:
        logger.warning(f"No observations found in baseline window {baseline_start_year}-{baseline_end_year}. Using all available years.")
        baseline_data = monthly_df[monthly_df["sst_mean_celsius"].notna()]

    climatology = {}
    for m in range(1, 13):
        m_subset = baseline_data[baseline_data["month"] == m]["sst_mean_celsius"]
        if len(m_subset) > 0:
            m_mean = float(m_subset.mean())
            m_std = float(m_subset.std(ddof=1)) if len(m_subset) > 1 else 0.5
            n_years = len(m_subset)
        else:
            logger.warning(f"Month {m} has 0 baseline samples. Defaulting to overall mean.")
            m_mean = float(baseline_data["sst_mean_celsius"].mean())
            m_std = 0.5
            n_years = 0

        climatology[m] = {
            "mean": round(m_mean, 4),
            "std": round(m_std, 4),
            "n_years": n_years
        }

    return climatology


def compute_sst_anomalies(
    monthly_df: pd.DataFrame,
    climatology: Dict[int, Dict[str, float]]
) -> pd.DataFrame:
    """
    Calculate monthly sea surface temperature anomalies relative to historical baseline.

    Formulas:
        Anomaly(y, m) = SST(y, m) - Baseline(m)
        Z(y, m) = Anomaly(y, m) / sigma(m)
    """
    df = monthly_df.copy()

    df["climatology_celsius"] = df["month"].map(lambda m: climatology[m]["mean"])
    df["climatology_std"] = df["month"].map(lambda m: climatology[m]["std"])

    # Monthly thermal anomaly (°C)
    df["anomaly_celsius"] = (df["sst_mean_celsius"] - df["climatology_celsius"]).round(4)

    # Standardized Z-Score
    df["z_score"] = np.where(
        df["climatology_std"] > 1e-4,
        (df["anomaly_celsius"] / df["climatology_std"]).round(3),
        0.0
    )

    return df


def detect_extreme_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Flag extreme marine temperature departures:
        - 'extreme_marine_heatwave': Z >= +2.0
        - 'extreme_upwelling_cooling': Z <= -2.0
        - 'normal': -2.0 < Z < +2.0
    """
    df = df.copy()

    conditions = [
        df["z_score"] >= z_threshold,
        df["z_score"] <= -z_threshold
    ]
    choices = [
        "extreme_marine_heatwave",
        "extreme_upwelling_cooling"
    ]

    df["anomaly_flag"] = np.select(conditions, choices, default="normal")
    return df
