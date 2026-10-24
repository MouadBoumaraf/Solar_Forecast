"""
Loading and cleaning the two raw data sources.

  PVGIS      hourly modelled energy of a 5 kWp system  -> the TARGET (what we predict)
  Open-Meteo hourly weather (ERA5)                     -> the INPUTS (what we predict from)

Both loaders check the file they are given, because a wrong file looks fine
until the results come out strange (we met every one of these problems).
"""
import re

import numpy as np
import pandas as pd
import pvlib

from config import (ELEV, LAT, LON, LOCATION_TOLERANCE_DEG, N_HOURS, PVGIS_FILE,
                    SUN_MIN_ELEVATION, SUNLIGHT_KEYWORDS, TARGET, WEATHER_FILE)


# ------------------------------------------------------------------ helpers
def find_table_start(path, max_lines=40):
    #i created this function to find the line number where the table starts in the CSV file. 
    #just to avoid always opening the csv file and counting the lines manually.
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("time,"):
                return i
            if i >= max_lines:
                break
    raise ValueError(f"{path.name}: no line starting with 'time,' found, so this is not a PVGIS/Open-Meteo CSV.")


def check_calendar(index, expected, name):
    #Checks if the index(time) of the table has any duplicated or missing hours.
    missing = expected.difference(index)
    if len(index) != len(expected) or len(missing) > 0 or index.duplicated().any():
        raise ValueError(f"{name}: expected {len(expected):,} distinct hours, got {len(index):,} "
                         f"({len(missing)} missing, {int(index.duplicated().sum())} duplicated).")


# -------------------------------------------------------------------- PVGIS
def load_pvgis_hourly(path=PVGIS_FILE):
    """Read the PVGIS file into an hourly table with a real time index (UTC)."""
    skip = find_table_start(path)
    # The file has a text footer, so we read exactly N_HOURS rows to stop before it.
    raw = pd.read_csv(path, skiprows=skip, nrows=N_HOURS)
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")   # text like 20050101:0010
    df = df.set_index("time").sort_index()
    # PVGIS stamps every hour at 10 minutes past
    check_calendar(df.index, pd.date_range("2005-01-01 00:10", "2023-12-31 23:10", freq="h"), "PVGIS")

    return df


def mark_gap_hours(df):
    #Flag hours where PVGIS wrote zeros instead of real data.
    #Some satellite data is missing, and the file fills it with 0. At night a zero is
    #real, but when the sun is clearly up it is impossible. We cannot trust the file's own
    #H_sun column here (it is zero inside the gaps too), so we calculate the true sun height
    #with pvlib and compare.
    df = df.copy()
    sun = pvlib.solarposition.get_solarposition(df.index.tz_localize("UTC"), LAT, LON, altitude=ELEV)
    df["sun_elevation"] = sun["apparent_elevation"].to_numpy()  
    sun_is_up = df["sun_elevation"] > SUN_MIN_ELEVATION
    df["gap"] = sun_is_up & (df["G(i)"] == 0) & (df["H_sun"] == 0)
    night_power = int(((df["sun_elevation"] <= 0) & (df["P"] > 0)).sum())
    return df


def pvgis_daily_energy(df):
    
    #P is in watts. One hour at P watts is P watt-hours, so a day's energy in kWh is the
    #sum of its hourly values divided by 1000. Days with gap hours become NaN ("unknown",
    #not zero) but stay in the table, so the calendar stays complete."""
    #i might in the future add a function to fill the gap hours with some kind of interpolation or prediction, 
    # but for now we just mark them as NaN.
    daily = pd.DataFrame({
        TARGET: df["P"].resample("D").sum() / 1000,
        "gap_hours": df["gap"].resample("D").sum(),
    })
    daily.loc[daily["gap_hours"] > 0, TARGET] = np.nan
    return daily



# --------------------------------------------------------------- Open-Meteo
def load_weather_hourly(path=WEATHER_FILE):
    meta = pd.read_csv(path, nrows=1)                 
    skip = find_table_start(path)
    w = pd.read_csv(path, skiprows=skip)
    w.columns = [re.sub(r"\s*\(.*\)\s*$", "", c).strip() for c in w.columns]   # 'temperature_2m (°C)' -> 'temperature_2m'
    w["time"] = pd.to_datetime(w["time"])              # text like 2005-01-01T00:00
    w = w.set_index("time").sort_index()
    check_calendar(w.index, pd.date_range("2005-01-01 00:00", "2023-12-31 23:00", freq="h"), "Open-Meteo")

    return w


def weather_daily_features(w):
   # sunlight columns : SUM of the hours / 1000  (kWh per m2 per day). Energy adds up over hours.
   # everything else  : AVERAGE of the 24 hours  (temperature, clouds, wind, humidity are levels).
   # Limit to remember: an average hides the timing of clouds, and it includes the night."""
    sunlight_cols = [c for c in w.columns if any(k in c for k in SUNLIGHT_KEYWORDS)]
    levels = w.drop(columns=sunlight_cols).resample("D").mean()
    sunlight = w[sunlight_cols].resample("D").sum() / 1000
    return levels.join(sunlight)
