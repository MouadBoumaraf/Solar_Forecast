
import logging
from datetime import date as Date, datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from Forecasting.open_meteo_client import WeatherAPIError
from Forecasting.solar_energy_predictor import SolarEnergyPredictor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Solar Energy Forecast API",
    description="Predict daily solar energy production in Hagenberg.",
    version="1.0.0",
)

predictor = SolarEnergyPredictor()
def parse_date(date_text: str) -> Date:
    try:
        requested_date = datetime.strptime(
            date_text, "%Y-%m-%d"
        ).date() #YYYY-MM-DD format
        if requested_date.isoformat() != date_text:
            raise ValueError()
        return requested_date
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format.Use YYYY-MM-DD."
        )
def validate_future_date(requested_date: Date):#Check if the date is old
    today = datetime.now(timezone.utc).date()
    if requested_date <= today:
        raise HTTPException(
            status_code=400,
            detail="Please select a future date. Past dates and today are not supported."
        )
def get_forecast_data() -> dict:
    try:
        return predictor.predict_next_days(days=7)
    except WeatherAPIError as exc:
        logger.exception("Weather request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve weather forecast data."
        ) from exc

def find_forecast(result: dict, requested_date: Date) -> dict:
    date_text = requested_date.isoformat()
    selected = next(
        (
            forecast
            for forecast in result["forecasts"]
            if forecast["date"] == date_text
        ),
        None,
    )
    if selected is None:
        available_dates = [
            forecast["date"]
            for forecast in result["forecasts"]
        ]
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Date is outside the available forecast range.",
                "available_dates": available_dates,
            },
        )
    return selected

def build_response(result: dict, selected: dict) -> dict:
    return {
        "model": result["model"],
        "unit": result["unit"],
        "date": selected["date"],
        "predicted_energy_kwh": selected["predicted_energy_kwh"],
        "comparison_source": result.get("comparison_source"),
        "comparison_status": result.get("comparison_status"),
        "forecast_solar_energy_kwh": selected.get(
            "forecast_solar_energy_kwh"
        ),
        "difference_kwh": selected.get("difference_kwh"),
    }

# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": predictor.model_name,
    }


# Main forecast endpoint
@app.get("/forecast")
def get_forecast(
    date: str = Query(
        ...,
        description="Date format: YYYY-MM-DD"
    )
):
    requested_date = parse_date(date)
    validate_future_date(requested_date)
    result = get_forecast_data()
    selected = find_forecast(result, requested_date)
    return build_response(result, selected)
