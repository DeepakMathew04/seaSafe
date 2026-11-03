"""
SEA-SAFE Analytics & Scientific Compute Package (Developer 2).

Provides peer-reviewed, reproducible oceanographic computation routines:
- temporal: Unit conversions (Kelvin -> Celsius exactly once), 2D spatial masking, monthly/seasonal aggregation, coverage calculation
- climatology: Baseline reference period (2003-2020) monthly normals, SST anomalies, standardized Z-scores
- autocorrelation: Autocorrelation Function (ACF), Hamed & Rao (1998) variance scaling
- trend: Theil-Sen median slopes, Mann-Kendall & Seasonal Mann-Kendall tests, 95% confidence intervals
- forecast_handler: Transparent forecast safety guardrail and climatological benchmarks
"""

from backend.analytics.temporal import (
    convert_kelvin_to_celsius,
    apply_spatial_mask,
    aggregate_daily_to_monthly,
    calculate_monthly_coverage,
    preprocess_and_aggregate
)
from backend.analytics.climatology import (
    calculate_monthly_climatology,
    compute_sst_anomalies,
    detect_extreme_anomalies
)
from backend.analytics.autocorrelation import (
    compute_autocorrelation_function,
    calculate_hamed_rao_variance_correction
)
from backend.analytics.trend import (
    calculate_theil_sen_trend,
    run_mann_kendall_test,
    run_seasonal_mann_kendall_test,
    compute_ols_baseline,
    calculate_complete_trend_analysis
)
from backend.analytics.forecast_handler import (
    get_forecast_guardrail_response,
    generate_climatological_forecast_benchmark
)

__all__ = [
    "convert_kelvin_to_celsius",
    "apply_spatial_mask",
    "aggregate_daily_to_monthly",
    "calculate_monthly_coverage",
    "preprocess_and_aggregate",
    "calculate_monthly_climatology",
    "compute_sst_anomalies",
    "detect_extreme_anomalies",
    "compute_autocorrelation_function",
    "calculate_hamed_rao_variance_correction",
    "calculate_theil_sen_trend",
    "run_mann_kendall_test",
    "run_seasonal_mann_kendall_test",
    "compute_ols_baseline",
    "calculate_complete_trend_analysis",
    "get_forecast_guardrail_response",
    "generate_climatological_forecast_benchmark"
]
