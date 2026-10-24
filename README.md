## How it works

1. The PVGIS hourly file is checked against the **true sun position** (pvlib). Hours where the file shows zero sunlight although
   the sun is clearly up are missing satellite data stored as zeros; those days are marked missing (266 days, 3.8%).
2. Hourly values become one row per day: energy = sum of the hours / 1000; sunlight is summed, other weather is averaged.
3. The table is split by time: **train 2005-2021, validation 2022, test 2023**.
4. Three baselines and three models are scored on the same validation days with MAE and RMSE.

## Run it

```
python -m venv .venv
.venv\Scripts\activate            (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# put the two raw files in data/raw/ (names are set in src/config.py)
python src/prepare_data.py        # step 1: raw files -> data/processed/
python src/run_evaluation.py      # step 2: baselines + models -> results/
```