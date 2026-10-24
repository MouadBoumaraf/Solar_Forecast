"""
The three machine-learning models.

  linear regression  - the simplest: a weighted sum of the inputs
  random forest      - many decision trees, their answers averaged
  gradient boosting  - trees built one after another, each fixing the previous mistakes

X = the inputs (weather + season), y = the answer (daily energy).
A model learns the link between X and y on the training days only, then predicts days it has never seen.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from config import RANDOM_STATE, TARGET

# Same list as in prepare_data.py: columns that would hand the answer to the model.
LEAKY_COLUMNS = {"P", "G(i)", "H_sun", "T2m", "WS10m", "irradiance_kwh_m2", "gap_hours"}


def add_calendar(table):
    """Add the time of year as two numbers.

    The year is treated like a clock face, so 31 December and 1 January end up next to each other.
    (A plain day number 1..365 would make them look 364 days apart.)"""
    table = table.copy()
    angle = 2 * np.pi * table.index.dayofyear / 365.25
    table["doy_sin"] = np.sin(angle)
    table["doy_cos"] = np.cos(angle)
    return table


def input_columns(table):
    """Every column except the answer. Stops if a leaky PVGIS column slipped in."""
    cols = [c for c in table.columns if c != TARGET]
    leaked = LEAKY_COLUMNS & set(cols)
    assert not leaked, f"leaky columns among the model inputs: {leaked}"
    return cols


def make_models():
    """Fresh, untrained models. random_state fixes the randomness so every run gives the same numbers."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def fit_and_predict(train, eval_days):
    """Train every model on `train`, predict `eval_days`.

    Returns (predictions, train_mae):
      predictions - DataFrame, one column per model (energy can't be negative, so we clip at 0)
      train_mae   - each model's error on its own training days (used to spot overfitting)"""
    train_m, eval_m = add_calendar(train), add_calendar(eval_days)
    cols = input_columns(train_m)
    X_train, y_train, X_eval = train_m[cols], train_m[TARGET], eval_m[cols]

    predictions, train_mae = {}, {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        predictions[name] = np.clip(model.predict(X_eval), 0, None)
        train_mae[name] = float(np.abs(np.clip(model.predict(X_train), 0, None) - y_train).mean())
    return pd.DataFrame(predictions, index=eval_days.index), train_mae
