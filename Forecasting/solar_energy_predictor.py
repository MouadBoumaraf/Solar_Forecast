
from pathlib import Path

import joblib
import numpy as np

from Training.config import PROJECT_ROOT
from Forecasting.open_meteo_client import fetch_hourly_forecast
from Forecasting.prepare_forecast_inputs import prepare_forecast_inputs
from Forecasting.forecast_solar_client import (
    fetch_forecast_solar_daily,
    ForecastSolarError,
)

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


class SolarEnergyPredictor:

    def __init__(self):
        # Load the trained model once
        artifact = joblib.load(MODEL_PATH)

        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.model_name = artifact.get("model_name", "unknown")

    def predict_next_days(self, days: int = 7) -> dict:
        hourly_weather = fetch_hourly_forecast(
            forecast_days=days + 1
        )
        daily_features = prepare_forecast_inputs(
            hourly_weather,
            days=days,
        )
        X = daily_features[self.feature_columns]
        #Predict daily energy production
        predictions = np.clip(
            self.model.predict(X),
            0,
            None,
        )
               # Attach each prediction to its date
        forecasts = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_energy_kwh": round(float(energy), 2),
            }
            for date, energy in zip(
                daily_features.index,
                predictions,
            )
        ]

        # Get the independent Forecast.Solar estimates
        try:
            reference = fetch_forecast_solar_daily()
            comparison_status = "available"
        except ForecastSolarError:
            reference = {}
            comparison_status = "unavailable"

        # Compare dates available from both services
        for forecast in forecasts:
            date = forecast["date"]
            solar_value = reference.get(date)

            forecast["forecast_solar_energy_kwh"] = (
                round(solar_value, 2)
                if solar_value is not None
                else None
            )

            forecast["difference_kwh"] = (
                round(
                    abs(
                        forecast["predicted_energy_kwh"]
                        - solar_value
                    ),
                    2,
                )
                if solar_value is not None
                else None
            )

        # Return the forecast and comparison
        return {
            "model": self.model_name,
            "unit": "kWh/day",
            "comparison_source": "Forecast.Solar",
            "comparison_status": comparison_status,
            "forecasts": forecasts,
        }