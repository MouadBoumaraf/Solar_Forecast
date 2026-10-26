from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import config
import models as model_module

MODEL_DIR = config.PROJECT_ROOT / "models"
RESULTS_DIR = config.RESULTS_DIR

def main():
    # 1. Load the daily dataset prepared by your existing pipeline
    data_path = config.PROCESSED_DIR / "daily_dataset.csv"

    dataset = pd.read_csv(
        data_path,
        index_col="date",
        parse_dates=True,
    ).sort_index()

    # Exclude 2023 before training or model selection.
    dataset = dataset[
        (dataset.index.year >= 2005)
        & (dataset.index.year <= config.VALID_YEAR)
    ].copy()

    usable = dataset.dropna()

    # 2. Preserve your existing chronological split
    train_raw = usable[
        usable.index.year <= config.TRAIN_END_YEAR
    ]

    valid_raw = usable[
        usable.index.year == config.VALID_YEAR
    ]
    #checking if train and valid datasets are empty
    if train_raw.empty or valid_raw.empty:
        raise ValueError(
            "Training or validation data is empty. "
            "Check your processed dataset and year settings."
        )
    train = model_module.add_calendar(train_raw)
    valid = model_module.add_calendar(valid_raw)

    feature_columns = model_module.input_columns(train)

    X_train = train[feature_columns]
    y_train = train[config.TARGET]

    X_valid = valid[feature_columns]
    y_valid = valid[config.TARGET]

    # 4. Tune each of your three existing models
    parameter_grids = {
        "linear_regression": {
            "fit_intercept": [True, False],
        },
        "random_forest": {
            "n_estimators": [200, 300],
            "max_depth": [None, 15],
            "min_samples_leaf": [1, 3],
        },
        "gradient_boosting": {
            "max_iter": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
        },
    }
    #find the best settings for each model (hyperparameter tuning) 
    cv = TimeSeriesSplit(n_splits=4)
    candidates = {}
    results = []

    for name, estimator in model_module.make_models().items():
        print(f"\nTuning {name}...")

       #avoid cpu overload for random forest by setting n_jobs to 1 instead of -1
        if name == "random_forest":
            estimator.set_params(n_jobs=1)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grids[name],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Evaluate on the same 2022 days for every model.
        predictions = np.clip(
            best_model.predict(X_valid), 0, None
        )

        mae = mean_absolute_error(y_valid, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_valid, predictions)
        )

        candidates[name] = {
            "model": best_model,
            "parameters": search.best_params_,
        }

        results.append({
            "model": name,
            "CV_MAE_kWh_day": -search.best_score_,
            "validation_MAE_kWh_day": mae,
            "validation_RMSE_kWh_day": rmse,
            "best_parameters": str(search.best_params_),
        })

    # 5. Compare the tuned models using 2022 MAE
    scores = pd.DataFrame(results).sort_values(
        "validation_MAE_kWh_day"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    scores.to_csv(
        RESULTS_DIR / "tuned_validation_scores.csv",
        index=False,
    )

    print("\nTuned model comparison on 2022:")
    print(scores.to_string(index=False))

    best_name = scores.iloc[0]["model"]
    print(f"\nPreferred model: {best_name}")

    # 6. Retrain each selected configuration on 2005–2022
    final_raw = usable[
        usable.index.year <= config.VALID_YEAR
    ]

    final_data = model_module.add_calendar(final_raw)
    X_final = final_data[feature_columns]
    y_final = final_data[config.TARGET]

    for name, candidate in candidates.items():
        final_model = candidate["model"]

        if name == "random_forest":
            final_model.set_params(n_jobs=-1)

        final_model.fit(X_final, y_final)

        artifact = {
            "model": final_model,
            "model_name": name,
            "feature_columns": feature_columns,
            "target_column": config.TARGET,
            "trained_until": "2022-12-31",
            "best_parameters": candidate["parameters"],
            "sklearn_version": sklearn.__version__,
        }

        joblib.dump(
            artifact, MODEL_DIR / f"{name}.joblib"
        )

        if name == best_name:
            joblib.dump(
                artifact, MODEL_DIR / "model.joblib"
            )

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Validation results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
