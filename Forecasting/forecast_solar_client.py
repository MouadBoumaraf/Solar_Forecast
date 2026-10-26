
import requests

from Training.config import LAT, LON


BASE_URL = "https://api.forecast.solar"

# Forecast.Solar uses a different azimuth convention
# from Open-Meteo: 0 degrees means south.
PANEL_TILT = 35
PANEL_AZIMUTH = 0
PANEL_CAPACITY_KWP = 5


class ForecastSolarError(RuntimeError):
    """Raised when Forecast.Solar cannot provide a forecast."""


def fetch_forecast_solar_daily() -> dict[str, float]:
    """Return Forecast.Solar daily energy estimates in kWh."""

    url = (
        f"{BASE_URL}/estimate/"
        f"{LAT}/{LON}/"
        f"{PANEL_TILT}/{PANEL_AZIMUTH}/"
        f"{PANEL_CAPACITY_KWP}"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError) as exc:
        raise ForecastSolarError(
            "Could not retrieve Forecast.Solar data."
        ) from exc

    daily_wh = data.get("result", {}).get("watt_hours_day")

    if not daily_wh:
        message = data.get("message", {})
        raise ForecastSolarError(
            message.get("text") or "No daily energy estimates returned."
        )

    # The API provides Wh/day; convert to kWh/day.
    return {
        date: float(energy_wh) / 1000
        for date, energy_wh in daily_wh.items()
    }
