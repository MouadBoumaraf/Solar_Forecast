
import pandas as pd
from Training.models import add_calendar
from Training.config import SUNLIGHT_KEYWORDS
from Forecasting.open_meteo_client import HOURLY_VARIABLES

#convert the data from hourly to daily and add calendar features
def prepare_forecast_inputs(
    hourly_data: pd.DataFrame,
    days: int = 7,
) -> pd.DataFrame:
    """Convert hourly weather into daily model features."""

    # 1. Select the next complete UTC days
    weather = hourly_data.copy()
    weather.index = pd.to_datetime(weather.index, utc=True)
    weather = weather.sort_index()

    start_day = (
        pd.Timestamp.now(tz="UTC").normalize()
        + pd.Timedelta(days=1)
    )
    dates = pd.date_range(start_day, periods=days, freq="D")
    end_day = start_day + pd.Timedelta(days=days)

    weather = weather.loc[
        (weather.index >= start_day)
        & (weather.index < end_day),
        HOURLY_VARIABLES,
    ]

    if not (weather.resample("D").size()
            .reindex(dates, fill_value=0) == 24).all():
        raise ValueError("Some forecast days are incomplete.")

    sunlight = [
        col for col in HOURLY_VARIABLES
        if any(key in col for key in SUNLIGHT_KEYWORDS)
    ]
    other = [col for col in HOURLY_VARIABLES if col not in sunlight]

    daily_weather = weather[other].resample("D").mean()
    daily_sunlight = weather[sunlight].resample("D").sum() / 1000

    daily_features = daily_weather.join(daily_sunlight)

    daily_features = add_calendar(daily_features)

    return daily_features
