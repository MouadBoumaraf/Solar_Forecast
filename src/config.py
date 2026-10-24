"""
Settings shared by every script in this project.

Change a value HERE and nowhere else. All paths are built from the project
root, so the scripts work no matter which folder you start them from.
"""
from pathlib import Path

# ----------------------------------------------------------------- folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent       
RAW_DIR = PROJECT_ROOT / "data" / "raw"                     
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"        
RESULTS_DIR = PROJECT_ROOT / "results"                     

# ------------------------------------------------------------ input files
PVGIS_FILE = RAW_DIR / "PVGIS.csv"                          
WEATHER_FILE = RAW_DIR / "OpenMeteo_2005_2023.csv"          

# --------------------------------------------------------- the PV system
LAT, LON, ELEV = 48.368, 14.512, 439                        # location used for the PVGIS download
N_HOURS = 166_536                                           # 19 years of hourly rows, 2005-2023
SUN_MIN_ELEVATION = 3                                       # degrees; sun higher than this + zero data = gap
LOCATION_TOLERANCE_DEG = 0.5                                # max distance between weather file and PVGIS point

# --------------------------------------------------------- time split
# Train on the early years, compare methods on the validation year,
# and keep the test year untouched until the very end.
TRAIN_END_YEAR = 2021                                       # training = 2005 ... 2021
VALID_YEAR = 2022
TEST_YEAR = 2023                                            # never loaded by run_evaluation.py

# ------------------------------------------------------------ naming
TARGET = "energy_kwh"                                       # the column the models must predict
SUNLIGHT_KEYWORDS = ("radiation", "irradiance")             # weather columns that are SUMMED per day

# ------------------------------------------------------- baseline settings
SMOOTHING_DAYS = 15                                         # +-7 days around each day of the year
CLEAR_DAY_QUANTILE = 0.9                                    # "sunny day" = 90th percentile of past energy
CLOUD_A, CLOUD_B = 0.75, 3.4                                # empirical cloud formula (Kasten-Czeplak); check before citing

# ------------------------------------------------------------- models
RANDOM_STATE = 0                                            # fixed seed: same result on every run
PLOT_MONTH = "2022-06"                                      # month shown in the day-by-day chart
