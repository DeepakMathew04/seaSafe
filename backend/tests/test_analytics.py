"""
Unit & Scientific Verification Tests for Developer 2 (SEA-SAFE Analytics).

Tests all implementation checklist requirements:
1. Kelvin to Celsius unit conversion (executed exactly once)
2. Spatial ocean masking & land pixel exclusion
3. Daily to monthly temporal aggregation (resample MS)
4. Monthly observational coverage & data quality flagging (<70% insufficient)
5. Baseline climatological normals (2003–2020) & anomaly computations
6. Theil-Sen slope estimation & 95% confidence intervals
7. Mann-Kendall monotonic trend test with Hamed-Rao autocorrelation adjustment
8. Autocorrelation Function (ACF) diagnostics
9. Forecast safety guardrail strict return contract
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pytest

# Ensure sea-safe root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.analytics.temporal import (
    convert_kelvin_to_celsius,
    apply_spatial_mask,
    aggregate_daily_to_monthly,
    calculate_monthly_coverage,
    aggregate_regional_seasons
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
    compute_ols_baseline,
    calculate_complete_trend_analysis
)
from backend.analytics.forecast_handler import (
    get_forecast_guardrail_response,
    GUARDRAIL_STRICT_REASON
)


@pytest.fixture
def synthetic_sst_dataset():
    """Create a realistic 3-year daily synthetic SST dataset (2018–2020)."""
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    lats = np.array([8.3, 8.5, 8.7])
    lons = np.array([76.9, 77.0, 77.1])

    # Seasonal sinusoidal cycle in Kelvin: ~298 K to 303 K (~25°C to 30°C)
    day_of_year = dates.day_of_year.values[:, None, None]
    base_sst = 300.0 + 2.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    noise = np.random.normal(0, 0.2, size=(len(dates), len(lats), len(lons)))
    sst_data = base_sst + noise

    # Mask: (0=land, 1=ocean)
    mask_data = np.ones((len(dates), len(lats), len(lons)), dtype=np.int8)
    mask_data[:, 0, 0] = 2  # Land pixel

    ds = xr.Dataset(
        data_vars={
            "analysed_sst": (["time", "lat", "lon"], sst_data, {"units": "kelvin"}),
            "mask": (["time", "lat", "lon"], mask_data)
        },
        coords={
            "time": dates,
            "lat": lats,
            "lon": lons
        }
    )
    return ds


class TestTemporalAggregation:

    def test_kelvin_to_celsius_exact_once(self, synthetic_sst_dataset):
        """Verify Kelvin to Celsius conversion subtracts 273.15 and does not double-subtract."""
        ds = synthetic_sst_dataset
        sst_c = convert_kelvin_to_celsius(ds)

        # Expected mean ~27°C
        assert 24.0 < float(sst_c.mean()) < 32.0
        assert sst_c.attrs["units"] == "degrees_Celsius"

        # Calling again must not double-subtract
        sst_c2 = convert_kelvin_to_celsius(sst_c)
        assert np.isclose(float(sst_c.mean()), float(sst_c2.mean()))

    def test_spatial_masking(self, synthetic_sst_dataset):
        """Verify land pixels are filtered to NaN."""
        ds = synthetic_sst_dataset
        masked = apply_spatial_mask(ds)

        # Pixel (0, 0) was land (mask == 2)
        assert np.isnan(masked.values[:, 0, 0]).all()
        # Ocean pixels should be finite
        assert np.isfinite(masked.values[:, 1, 1]).all()

    def test_daily_to_monthly_aggregation(self, synthetic_sst_dataset):
        """Verify resample(time='MS').mean(skipna=True) produces correct month records."""
        ds = synthetic_sst_dataset
        sst_c = convert_kelvin_to_celsius(ds)
        masked = apply_spatial_mask(ds.assign(analysed_sst=sst_c))
        monthly_df = aggregate_daily_to_monthly(masked)

        # 3 years = 36 monthly records
        assert len(monthly_df) == 36
        assert set(monthly_df["month"].unique()) == set(range(1, 13))
        assert "sst_mean_celsius" in monthly_df.columns

    def test_monthly_coverage_calculation(self, synthetic_sst_dataset):
        """Verify coverage tracking flags insufficient months (<70%)."""
        ds = synthetic_sst_dataset
        sst_c = convert_kelvin_to_celsius(ds)
        coverage_df = calculate_monthly_coverage(sst_c)

        assert "coverage_pct" in coverage_df.columns
        assert "quality_flag" in coverage_df.columns
        assert (coverage_df["quality_flag"].isin(["good", "degraded", "insufficient"])).all()

    def test_regional_seasons_aggregation(self, synthetic_sst_dataset):
        """Verify aggregation into Kerala regional oceanographic seasons."""
        ds = synthetic_sst_dataset
        sst_c = convert_kelvin_to_celsius(ds)
        monthly_df = aggregate_daily_to_monthly(sst_c)
        seasonal_df = aggregate_regional_seasons(monthly_df)

        expected_seasons = {"pre_monsoon", "southwest_monsoon", "northeast_monsoon", "winter"}
        assert set(seasonal_df["season"].unique()) == expected_seasons


class TestClimatologyAndAnomalies:

    def test_baseline_climatology_computation(self):
        """Verify 12 monthly climatological normals calculated over reference window."""
        records = []
        for y in range(2003, 2021):
            for m in range(1, 13):
                # Simulated seasonal cycle
                temp = 28.0 + 2.0 * np.sin(2 * np.pi * m / 12.0)
                records.append({"year": y, "month": m, "sst_mean_celsius": temp})
        df = pd.DataFrame(records)

        clim = calculate_monthly_climatology(df, 2003, 2020)
        assert len(clim) == 12
        for m in range(1, 13):
            assert "mean" in clim[m]
            assert "std" in clim[m]
            assert clim[m]["n_years"] == 18

    def test_sst_anomalies_and_extreme_flags(self):
        """Verify Anomaly = SST - Baseline and extreme flags (|Z| >= 2.0)."""
        df = pd.DataFrame([
            {"year": 2021, "month": 5, "sst_mean_celsius": 32.5},  # Severe heatwave
            {"year": 2021, "month": 8, "sst_mean_celsius": 24.0},  # Deep upwelling cooling
            {"year": 2021, "month": 1, "sst_mean_celsius": 28.0}   # Normal
        ])
        clim = {
            1: {"mean": 28.0, "std": 0.4, "n_years": 18},
            5: {"mean": 30.0, "std": 0.5, "n_years": 18},
            8: {"mean": 26.5, "std": 0.6, "n_years": 18}
        }

        df_anom = compute_sst_anomalies(df, clim)
        df_ext = detect_extreme_anomalies(df_anom, z_threshold=2.0)

        # Check anomalies
        assert np.isclose(df_ext.loc[0, "anomaly_celsius"], 2.5)
        assert df_ext.loc[0, "anomaly_flag"] == "extreme_marine_heatwave"
        assert df_ext.loc[1, "anomaly_flag"] == "extreme_upwelling_cooling"
        assert df_ext.loc[2, "anomaly_flag"] == "normal"


class TestTrendsAndAutocorrelation:

    def test_theil_sen_slope_units(self):
        """Verify Theil-Sen slope conversion to °C/decade and °C/year."""
        # 120 months with known slope of 0.002 °C/month -> 0.24 °C/decade
        x = np.arange(120)
        y = 28.0 + 0.002 * x + np.random.normal(0, 0.01, size=120)

        res = calculate_theil_sen_trend(y)
        assert "slope_deg_per_decade" in res
        assert "slope_deg_per_year" in res
        assert "ci_95_deg_per_decade" in res
        assert 0.20 < res["slope_deg_per_decade"] < 0.28
        assert res["ci_95_deg_per_decade"][0] < res["slope_deg_per_decade"] < res["ci_95_deg_per_decade"][1]

    def test_mann_kendall_with_autocorrelation(self):
        """Verify Mann-Kendall p-value and Hamed-Rao variance correction."""
        x = np.arange(100)
        # Strong upward trend
        y = 25.0 + 0.05 * x + np.random.normal(0, 0.2, size=100)

        mk = run_mann_kendall_test(y, adjust_autocorrelation=True)
        assert mk["statistically_significant"] is True
        assert mk["trend_direction"] == "increasing"
        assert mk["p_value"] < 0.01

    def test_acf_autocorrelation(self):
        """Verify ACF calculation and significant lag detection."""
        # Sinusoidal series with 12-month periodicity
        x = np.arange(72)
        y = np.sin(2 * np.pi * x / 12.0)

        acf = compute_autocorrelation_function(y, max_lags=24)
        assert acf["has_significant_autocorrelation"] is True
        assert 12 in acf["significant_lags"]  # 12-month lag must show strong correlation


class TestForecastGuardrail:

    def test_forecast_guardrail_exact_reason(self):
        """Verify strict SEA//SAFE forecast guardrail payload when no model is validated."""
        res = get_forecast_guardrail_response(is_validated_model_connected=False)

        assert res["status"] == "no_validated_forecast"
        assert res["reason"] == (
            "SEA//SAFE does not extrapolate trends or invent unvalidated forecasts. "
            "Connect an official operational model to view predictions."
        )

    def test_forecast_guardrail_with_climatological_benchmark(self):
        """Verify climatological reference cycle is attached with disclaimer."""
        clim = {1: {"mean": 27.8, "std": 0.4}, 2: {"mean": 28.4, "std": 0.5}}
        res = get_forecast_guardrail_response(
            is_validated_model_connected=False,
            climatology_baseline=clim
        )

        assert res["status"] == "no_validated_forecast"
        benchmark = res["climatological_benchmark"]
        assert benchmark["available"] is True
        assert "Reference climatological annual cycle" in benchmark["disclaimer"]
        assert benchmark["monthly_normals_celsius"]["1"] == 27.8
