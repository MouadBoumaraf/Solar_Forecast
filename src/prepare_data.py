"""
STEP 1 - raw files -> clean daily tables.

Run from anywhere:   python src/prepare_data.py

What it does
  1. loads the PVGIS file, finds the gap hours, builds the daily energy (the TARGET)
  2. loads the Open-Meteo file and builds the daily weather inputs
  3. joins both by date, checks the result, splits it by time and saves it

Output (data/processed/):
  daily_dataset.csv   all days (gap days keep an empty energy_kwh, so the calendar is complete)
  train.csv           usable days up to TRAIN_END_YEAR
  valid.csv           usable days of VALID_YEAR
  test.csv            usable days of TEST_YEAR  (do not open until the very end)
"""
import pandas as pd
import data_loading as dl
from config import PROCESSED_DIR, TARGET, TEST_YEAR, TRAIN_END_YEAR, VALID_YEAR



def build_dataset():
    """Return (dataset, hourly_pvgis). dataset = one row per day: energy + weather inputs."""
    pvgis = dl.mark_gap_hours(dl.load_pvgis_hourly())
    energy = dl.pvgis_daily_energy(pvgis)
    weather = dl.weather_daily_features(dl.load_weather_hourly())
    dataset = energy[[TARGET]].join(weather, how="left")
    dataset.index.name = "date"
    return dataset, pvgis

def main():
    dataset, pvgis = build_dataset()
    usable = dataset.dropna() # removes the gap days (no known energy)
    train = usable.loc[:str(TRAIN_END_YEAR)]
    valid = usable.loc[str(VALID_YEAR)]
    test = usable.loc[str(TEST_YEAR)]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DIR / "daily_dataset.csv", index_label="date")
    train.to_csv(PROCESSED_DIR / "train.csv", index_label="date")
    valid.to_csv(PROCESSED_DIR / "valid.csv", index_label="date")
    test.to_csv(PROCESSED_DIR / "test.csv", index_label="date")
    gap_days = int(dataset[TARGET].isna().sum())
    print(f"split: train {len(train)} | valid {len(valid)} | test {len(test)} days (gap days excluded)")
    print(f"saved in {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
