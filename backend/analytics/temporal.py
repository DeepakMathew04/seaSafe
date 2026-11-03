"""
Temporal & Spatial Aggregation Module (Developer 2).

Scope & Responsibilities:
1. Unit Conversion: Convert temperature from Kelvin to Celsius: T(°C) = T(K) - 273.15.
   Ensures this is executed exactly once by verifying variable values and units attributes.
2. 2D Spatial Masking: Applies Developer 3's ocean mask so that land and mixed pixels are excluded.
3. Daily to Monthly Aggregation: resample(time="MS").mean(skipna=True).
4. Coverage Tracking: Calculates valid observational coverage per month, flagging insufficient data (<70%).
5. Regional Kerala Oceanographic Seasons: Pre-Monsoon (MAM), SW Monsoon (JJAS), NE Monsoon (OND), Winter (JF).
"""

import logging
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("SeaSafe.Temporal")


def convert_kelvin_to_celsius(da_or_ds: Union[xr.DataArray, xr.Dataset], var_name: str = "analysed_sst") -> xr.DataArray:
    """
    Convert gridded SST data from Kelvin to Celsius exactly once.

    Formula:
        T(°C) = T(K) - 273.15

    Safety Checks:
        - Inspects dataset metadata units (e.g. 'kelvin', 'K').
        - Inspects value ranges: If mean SST is already < 100.0, it is already Celsius and conversion is skipped.
        - Prevents accidental double-subtraction.
    """
    if isinstance(da_or_ds, xr.Dataset):
        if var_name not in da_or_ds.data_vars:
            raise KeyError(f"Variable '{var_name}' not found in dataset. Variables: {list(da_or_ds.data_vars)}")
        da = da_or_ds[var_name]
    else:
        da = da_or_ds

    # Check units attribute
    units_attr = str(da.attrs.get("units", "")).lower()

    # Sample non-NaN values to verify scale
    valid_sample = da.values[np.isfinite(da.values)]
    if len(valid_sample) == 0:
        logger.warning("Empty or all-NaN SST slice passed to convert_kelvin_to_celsius.")
        return da

    sample_mean = float(np.nanmean(valid_sample[:1000]))

    # If already Celsius (~15°C to 35°C for tropical oceans)
    if sample_mean < 100.0 or "celsius" in units_attr or units_attr in ["c", "degc", "degrees_c"]:
        logger.info(f"SST data is already in Celsius (mean={sample_mean:.2f}°C). Skipping subtraction.")
        da_celsius = da.copy()
        da_celsius.attrs["units"] = "degrees_Celsius"
        return da_celsius

    # Convert Kelvin (expected ~295K to 305K) to Celsius
    logger.info(f"Converting SST from Kelvin to Celsius (sample mean={sample_mean:.2f} K).")
    da_celsius = da - 273.15
    da_celsius.attrs["units"] = "degrees_Celsius"
    da_celsius.attrs["conversion"] = "Subtracted 273.15 from Kelvin"

    return da_celsius


def apply_spatial_mask(
    ds: xr.Dataset,
    ocean_mask_2d: Optional[np.ndarray] = None,
    var_name: str = "analysed_sst",
    mask_var_name: str = "mask"
) -> xr.DataArray:
    """
    Apply Developer 3's 2D ocean mask and MUR dataset flags to isolate open water pixels.

    Filters:
        1. MUR Level-4 Quality Mask: mask == 1 is open ocean (mask == 2 is land).
        2. Developer 3 Offshore Buffer Mask: 2D boolean array (True = valid sea within buffer).
        3. Excludes fill values (-32768, NaN, Inf).
    """
    sst = ds[var_name]

    # Step 1: MUR Native Mask filtering
    if mask_var_name in ds.data_vars:
        mur_ocean = ds[mask_var_name] == 1
        sst = sst.where(mur_ocean)

    # Step 2: Developer 3 Offshore Marine Buffer Mask (2D lat/lon)
    if ocean_mask_2d is not None:
        ocean_mask_bool = np.asarray(ocean_mask_2d, dtype=bool)
        if ocean_mask_bool.shape != (len(ds.lat), len(ds.lon)):
            raise ValueError(
                f"Ocean mask shape {ocean_mask_bool.shape} does not match "
                f"grid dimensions (lat: {len(ds.lat)}, lon: {len(ds.lon)})"
            )

        mask_da = xr.DataArray(
            ocean_mask_bool,
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"]
        )
        sst = sst.where(mask_da)

    return sst


def aggregate_daily_to_monthly(daily_sst: xr.DataArray) -> pd.DataFrame:
    """
    Aggregate daily gridded SST observations to monthly means.

    Uses:
        resample(time="MS").mean(skipna=True)
    """
    # Spatial mean for each day over valid ocean pixels
    spatial_daily = daily_sst.mean(dim=["lat", "lon"], skipna=True)

    # Convert to xarray time-series and resample to Month-Start (MS)
    monthly_xr = spatial_daily.resample(time="MS").mean(skipna=True)
    monthly_df = monthly_xr.to_dataframe(name="sst_mean_celsius").reset_index()

    monthly_df["year"] = monthly_df["time"].dt.year
    monthly_df["month"] = monthly_df["time"].dt.month
    monthly_df["date"] = monthly_df["time"].dt.strftime("%Y-%m")

    return monthly_df


def calculate_monthly_coverage(
    daily_sst: xr.DataArray,
    min_coverage_pct: float = 70.0
) -> pd.DataFrame:
    """
    Calculate observational data coverage per month.

    Coverage Metric:
        coverage_pct = (valid observed pixel-days) / (total theoretical pixel-days) * 100.0

    Flags:
        - 'good': coverage >= 90%
        - 'degraded': 70% <= coverage < 90%
        - 'insufficient': coverage < 70%
    """
    total_pixels_per_day = daily_sst.sizes["lat"] * daily_sst.sizes["lon"]

    # Count valid (non-NaN) pixels per day
    valid_daily = np.isfinite(daily_sst).sum(dim=["lat", "lon"])

    # Aggregate by month
    monthly_valid = valid_daily.resample(time="MS").sum()
    monthly_days = valid_daily.resample(time="MS").count()

    monthly_coverage_pct = (monthly_valid / (monthly_days * total_pixels_per_day)) * 100.0
    coverage_df = monthly_coverage_pct.to_dataframe(name="coverage_pct").reset_index()

    def assign_quality_flag(pct):
        if np.isnan(pct) or pct < min_coverage_pct:
            return "insufficient"
        elif pct >= 90.0:
            return "good"
        else:
            return "degraded"

    coverage_df["quality_flag"] = coverage_df["coverage_pct"].apply(assign_quality_flag)
    coverage_df["valid_pixels"] = monthly_valid.values

    return coverage_df


def aggregate_regional_seasons(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly records into regional Kerala oceanographic seasons:
    - Pre-Monsoon (MAM: March, April, May)
    - Southwest Monsoon (JJAS: June, July, August, September)
    - Northeast Monsoon (OND: October, November, December)
    - Winter (JF: January, February)
    """
    season_map = {
        1: ("winter", [1, 2]),
        2: ("winter", [1, 2]),
        3: ("pre_monsoon", [3, 4, 5]),
        4: ("pre_monsoon", [3, 4, 5]),
        5: ("pre_monsoon", [3, 4, 5]),
        6: ("southwest_monsoon", [6, 7, 8, 9]),
        7: ("southwest_monsoon", [6, 7, 8, 9]),
        8: ("southwest_monsoon", [6, 7, 8, 9]),
        9: ("southwest_monsoon", [6, 7, 8, 9]),
        10: ("northeast_monsoon", [10, 11, 12]),
        11: ("northeast_monsoon", [10, 11, 12]),
        12: ("northeast_monsoon", [10, 11, 12])
    }

    df = monthly_df.copy()
    df["season"] = df["month"].map(lambda m: season_map[m][0])

    seasonal_records = []
    for (year, season), group in df.groupby(["year", "season"]):
        months_inc = sorted(group["month"].unique().tolist())
        seasonal_records.append({
            "year": int(year),
            "season": str(season),
            "sst_mean_celsius": float(group["sst_mean_celsius"].mean()),
            "months_included": months_inc
        })

    return pd.DataFrame(seasonal_records)


def preprocess_and_aggregate(
    ds: xr.Dataset,
    ocean_mask_2d: Optional[np.ndarray] = None,
    var_name: str = "analysed_sst"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete temporal and spatial preprocessing pipeline:
    1. Converts Kelvin to Celsius (exactly once).
    2. Applies ocean quality mask and 2D spatial buffer.
    3. Aggregates to monthly means.
    4. Computes coverage percentages and flags.
    5. Computes regional seasonal averages.

    Returns:
        (monthly_df, seasonal_df)
    """
    # 1. Kelvin -> Celsius (done once)
    sst_celsius = convert_kelvin_to_celsius(ds, var_name=var_name)
    ds_celsius = ds.copy()
    ds_celsius[var_name] = sst_celsius

    # 2. 2D Spatial Masking
    masked_sst = apply_spatial_mask(ds_celsius, ocean_mask_2d=ocean_mask_2d, var_name=var_name)

    # 3. Daily -> Monthly Aggregation
    monthly_df = aggregate_daily_to_monthly(masked_sst)

    # 4. Coverage Tracking
    coverage_df = calculate_monthly_coverage(masked_sst)
    monthly_df["coverage_pct"] = coverage_df["coverage_pct"].round(2)
    monthly_df["quality_flag"] = coverage_df["quality_flag"]
    monthly_df["valid_pixels"] = coverage_df["valid_pixels"].astype(int)

    # 5. Seasonal Aggregation
    seasonal_df = aggregate_regional_seasons(monthly_df)

    return monthly_df, seasonal_df
