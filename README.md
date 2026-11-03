# SEA//SAFE — Coastal Ocean Thermal Monitoring & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Developer 2 Status](https://img.shields.io/badge/Developer%202%20Analytics-Validated%20(12%2F12%20Tests)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> High-resolution satellite Sea Surface Temperature (SST) analytics, non-parametric trend quantification, climatological anomaly tracking, and responsible coastal forecast guardrails for the Thiruvananthapuram and southwest Indian coastline.

---

## 🗺️ Project Architecture & Developer Responsibilities

The SEA//SAFE codebase is organized across six specialized engineering roles as mapped in the **SEA-SAFE Project Structure Infographic** (`docs/SEA-SAFE Project Structure Infographic.png`):

| Module | Role | Status | Directory |
|---|---|---|---|
| **Developer 1** | NASA Data Ingestion & ETL (PO.DAAC/CMR) | Scaffolded | `backend/pipeline/` |
| **Developer 2** | Scientific Compute & Analytics Engineer | **Fully Implemented & Tested** | `backend/analytics/` |
| **Developer 3** | Geospatial & Boundary Engineer (5/10/20km Buffers) | Scaffolded | `backend/geospatial/` |
| **Developer 4** | Backend API & Service Engineer (FastAPI/Pydantic) | Scaffolded | `backend/api/` |
| **Developer 5** | Frontend Core Developer (Next.js / Tailwind) | Scaffolded | `frontend/src/app/`, `frontend/src/components/` |
| **Developer 6** | Frontend Geospatial & Data Viz (MapLibre/Charts) | Scaffolded | `frontend/src/components/visualization/` |

---

## 🔬 Developer 2: Scientific Compute & Analytics Suite

Developer 2's scientific core is fully implemented in `backend/analytics/` adhering to peer-reviewed oceanographic standards:

1. **Unit Conversions & Masking Safeguards (`backend/analytics/temporal.py`):**
   - Single-pass Kelvin to Celsius conversion ($T_{\text{Celsius}} = T_{\text{Kelvin}} - 273.15$) with metadata inspection and threshold sanity safeguards ($T < 100^\circ\text{C}$).
   - 2D Boolean ocean masking excluding land and mixed pixels (`mask == 1` open ocean).
   - Daily to monthly temporal aggregation (`resample(time="MS").mean(skipna=True)`).
   - Data coverage tracking with quality categorization ($\ge 90\%$ good, $70\%-89\%$ degraded, $<70\%$ insufficient).

2. **Climatological Baseline & Anomalies (`backend/analytics/climatology.py`):**
   - Fixed 18-year reference baseline (2003-01-01 to 2020-12-31):
     $$\text{Baseline}(m) = \frac{1}{N} \sum_{y=1}^N \text{SST}(y, m)$$
   - Monthly anomaly:
     $$\text{Anomaly}(y, m) = \text{SST}(y, m) - \text{Baseline}(m)$$
   - Standardized Z-Score:
     $$Z(y, m) = \frac{\text{Anomaly}(y, m)}{\sigma_m}$$
   - Marine heatwave identification ($Z \ge +2.0$) and upwelling anomaly detection ($Z \le -2.0$).

3. **Non-Parametric Trend Estimation (`backend/analytics/trend.py` & `autocorrelation.py`):**
   - Theil-Sen non-parametric median pairwise slope ($\beta_{\text{Sen}}$) and 95% rank confidence interval.
   - Standard scientific reporting units: **$^\circ\text{C}/\text{decade}$** ($\text{slope} \times 120$) and **$^\circ\text{C}/\text{year}$** ($\text{slope} \times 12$).
   - Mann-Kendall rank monotonic trend test ($S$ statistic, tie adjustment).
   - Autocorrelation Function (ACF) over lags 1–24 with Bartlett 95% confidence bounds ($\pm 1.96/\sqrt{N}$).
   - Hamed & Rao (1998) variance correction factor ($\eta$) applied to $V(S)$ to eliminate false-positive trend significance from serial correlation.

4. **Responsible Forecast Safety Guardrail (`backend/analytics/forecast_handler.py`):**
   - Strictly enforces the unvalidated forecast guardrail:
     ```json
     {
       "status": "no_validated_forecast",
       "reason": "SEA//SAFE does not extrapolate trends or invent unvalidated forecasts. Connect an official operational model to view predictions."
     }
     ```

---

## 🧪 Developer 2 Verification & Unit Testing

To run Developer 2's scientific test suite:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Developer 2 test suite
pytest tests/test_analytics.py -v
```

All 12 unit tests validate the complete checklist with 100% pass rate:
- Single-pass Kelvin to Celsius conversion
- Idempotency / threshold protections against double-subtraction
- 2D ocean masking application
- Daily-to-monthly aggregation and coverage flagging
- 18-year climatological baseline normals (2003–2020)
- Anomaly and standardized Z-Score computations
- Theil-Sen slope calculation and 95% confidence intervals
- Slope unit conversions (°C/decade and °C/year)
- Mann-Kendall test with Hamed-Rao serial correlation adjustment
- Autocorrelation Function (ACF) bounds
- Seasonal Mann-Kendall test across 12 monthly strata
- Strict forecast guardrail payload response

---

## 🖼️ Architecture Infographic Reference

The architecture infographic file provided for the project is located at:
- `docs/SEA-SAFE Project Structure Infographic.png`
- `frontend/public/images/SEA-SAFE Project Structure Infographic.png`
