"""
Forecast Guardrail & Outlook Handler (Developer 2).

Scope & Responsibilities:
1. Responsible Scientific Guardrail:
   If no validated forecast model is connected, explicitly return the strict safety payload:
   {
     "status": "no_validated_forecast",
     "reason": "SEA//SAFE does not extrapolate trends or invent unvalidated forecasts. Connect an official operational model to view predictions."
   }
2. Transparent Climatological Reference Benchmark:
   Attaches the historical 2003–2020 annual cycle as a non-predictive baseline with explicit disclaimers.
3. Operational Model Connector:
   Formats predictions when an externally validated high-resolution model (e.g. INCOIS/HOOFS or ML ensemble) is active.
"""

from typing import Dict, Any, Optional

GUARDRAIL_STRICT_REASON = (
    "SEA//SAFE does not extrapolate trends or invent unvalidated forecasts. "
    "Connect an official operational model to view predictions."
)


def get_forecast_guardrail_response(
    is_validated_model_connected: bool = False,
    validated_forecast_data: Optional[Dict[str, Any]] = None,
    climatology_baseline: Optional[Dict[int, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Generate forecast response conforming strictly to SEA//SAFE scientific guardrails.

    Returns:
        Structured dictionary matching frontend and backend API contracts.
    """
    if not is_validated_model_connected or not validated_forecast_data:
        benchmark = None
        if climatology_baseline:
            benchmark = {
                "available": True,
                "disclaimer": (
                    "Reference climatological annual cycle derived from historical 2003-2020 observations. "
                    "This represents historical averages, NOT a predictive operational forecast."
                ),
                "monthly_normals_celsius": {
                    str(m): info["mean"] for m, info in climatology_baseline.items()
                }
            }

        return {
            "status": "no_validated_forecast",
            "reason": GUARDRAIL_STRICT_REASON,
            "climatological_benchmark": benchmark
        }

    # When validated forecast model is connected
    return {
        "status": "validated_forecast",
        "reason": "Operational regional oceanographic model active.",
        "forecast_data": validated_forecast_data
    }


def generate_climatological_forecast_benchmark(climatology: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """Format climatology as a transparent benchmark for UI rendering."""
    return {
        "available": True,
        "disclaimer": "Reference climatological annual cycle derived from historical 2003-2020 observations.",
        "monthly_normals_celsius": {str(m): info["mean"] for m, info in climatology.items()}
    }
