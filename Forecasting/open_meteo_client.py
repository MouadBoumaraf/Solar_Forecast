
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from Training.config import ELEV, LAT, LON, PROJECT_ROOT


API_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
]

CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

cache_session = requests_cache.CachedSession(
    str(CACHE_DIR / "open_meteo"),
    expire_after=3600,
)

retry_session = retry(
    cache_session,
    retries=5,
    backoff_factor=0.2,
)

openmeteo = openmeteo_requests.Client(
    session=retry_session
)


class WeatherAPIError(RuntimeError):
    """Raised when weather data cannot be retrieved or processed."""

# Build the request parameters

def _build_params(forecast_days: int) -> dict:
    """Create the parameters sent to Open-Meteo."""

    return {
        "latitude": LAT,
        "longitude": LON,
        "elevation": ELEV,
        "hourly": HOURLY_VARIABLES,
        "forecast_days": forecast_days,
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "tilt": 35,
        "azimuth": 180,
    }


# Convert hourly variables into a dictionary 

def _extract_hourly_data(hourly) -> dict:
    """Extract each requested weather variable as a NumPy array."""

    data = {}

    for index, variable_name in enumerate(HOURLY_VARIABLES):
        variable = hourly.Variables(index)
        data[variable_name] = variable.ValuesAsNumpy()

    return data


# Create timestamps for the hourly observations

def _create_timestamps(hourly, number_of_rows: int) -> pd.DatetimeIndex:
    """Generate one UTC timestamp for each hourly observation."""

    start_time = pd.to_datetime(
        hourly.Time(),
        unit="s",
        utc=True,
    )

    interval = pd.Timedelta(
        seconds=hourly.Interval()
    )

    return pd.date_range(
        start=start_time,
        periods=number_of_rows,
        freq=interval,
    )


# Build the weather DataFrame 

def _build_weather_dataframe(hourly) -> pd.DataFrame:
    """Combine weather arrays and timestamps into one DataFrame."""

    data = _extract_hourly_data(hourly)

    # Use the first variable to determine the number of observations.
    number_of_rows = len(data[HOURLY_VARIABLES[0]])

    # Every variable must have one value per timestamp.
    for variable_name, values in data.items():
        if len(values) != number_of_rows:
            raise WeatherAPIError(
                f"Inconsistent number of values for {variable_name}."
            )

    timestamps = _create_timestamps(
        hourly,
        number_of_rows,
    )

    weather = pd.DataFrame(
        data=data,
        index=timestamps,
    )

    weather.index.name = "date"

    return weather


#  the main  function: fetch the weather 

def fetch_hourly_forecast(
    forecast_days: int = 8,
) -> pd.DataFrame:
    
    """Fetch hourly weather and return a UTC-indexed DataFrame."""

    if not 1 <= forecast_days <= 16:
        raise ValueError(
            "forecast_days must be between 1 and 16."
        )

    params = _build_params(forecast_days)

    try:
        responses = openmeteo.weather_api(
            API_URL,
            params=params,
        )

        if not responses:
            raise WeatherAPIError(
                "Open-Meteo returned no locations."
            )

        response = responses[0]
        hourly = response.Hourly()

        if hourly is None:
            raise WeatherAPIError(
                "Open-Meteo returned no hourly data."
            )

        return _build_weather_dataframe(hourly)

    except WeatherAPIError:
        raise

    except Exception as exc:
        raise WeatherAPIError(
            "Failed to retrieve or process Open-Meteo data."
        ) from exc
