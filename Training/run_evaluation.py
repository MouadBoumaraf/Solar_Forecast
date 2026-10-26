import pandas as pd
import baselines 
import config
import evaluation
import models


def main():
    dataset = pd.read_csv(config.PROCESSED_DIR / "daily_dataset.csv", index_col="date", parse_dates=True)
    dataset = dataset.loc[:str(config.VALID_YEAR)]                    # drop the test year right away

    usable = dataset.dropna()                                  # days with a known energy value
    train = usable.loc[:str(config.TRAIN_END_YEAR)]
    valid = usable.loc[str(config.VALID_YEAR)]

    # one table: the real value plus every method's prediction for each validation day
    preds = pd.DataFrame({"actual": valid[config.TARGET]})
    preds = preds.join(baselines.all_baselines(dataset, train, valid))
    model_preds, train_mae = models.fit_and_predict(train, valid)
    preds = preds.join(model_preds)

    scores, scored = evaluation.score_table(preds)
    print(scores.to_string())
    print(f"\nbest baseline: {scores.attrs['best_baseline']}")

    print("\ntraining error vs validation error (MAE, kWh):")
    for name, tr in train_mae.items():
        print(f"  {name:18s} train {tr:.2f} | valid {scores.loc[name, 'MAE_kwh']:.2f}")

    evaluation.save_results(scores, scored)
    evaluation.plot_comparison(scores, preds)
    print("\nsaved results in the results/ folder")


if __name__ == "__main__":
    main()
