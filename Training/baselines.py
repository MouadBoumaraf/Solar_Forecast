import numpy as np
import pandas as pd

from config import CLEAR_DAY_QUANTILE, CLOUD_A, CLOUD_B, SMOOTHING_DAYS, TARGET


def day_of_year(index):
    #Day number in the year, 1..365 
    return np.minimum(index.dayofyear, 365)


def day_smoother(by_day):
    #Gets the mean of pd series to make the data more accurate (by considering the fluxiation in energy in different days)
    return by_day.rolling(SMOOTHING_DAYS, center=True, min_periods=1).mean()


def same_as_yesterday(dataset, eval_days):
    #Baseline 1: tomorrow produces what today produced.
    return dataset[TARGET].shift(1).reindex(eval_days.index)


def seasonal_average(train, eval_days):
    #Baseline 2: the average energy of this time of year in the training years.(using Day_smoother to make it more accurate)
    by_day = train.groupby(day_of_year(train.index))[TARGET].mean()
    return day_smoother(by_day).reindex(day_of_year(eval_days.index)).to_numpy()
    #train.groupby(day_of_year(train.index)) returns table with number of years * rows of each day for example
    #2005/1/1 =>1 1 2006/1/1 => 1 ... 2005/1/2 => 2 
    #then get the target mean 

def cloud_rule(train, eval_days):
    #Baseline 3: sunny-day output for this time of year, reduced by the day's cloud cover.
    #sunny-day output = 90th percentile of the training energy on that day of the year
    #reduction ;= 1 - 0.75 * cloud_fraction ** 3.4 """
    by_day = train.groupby(day_of_year(train.index))[TARGET].quantile(CLEAR_DAY_QUANTILE)
    clear_day = day_smoother(by_day).reindex(day_of_year(eval_days.index)).to_numpy()
    cloud_fraction = eval_days["cloud_cover"] / 100                  # 0 = clear sky, 1 = overcast
    return clear_day * (1 - CLOUD_A * cloud_fraction ** CLOUD_B)

def all_baselines(dataset, train, eval_days):
    return pd.DataFrame({
        "baseline_1": same_as_yesterday(dataset, eval_days),
        "baseline_2": seasonal_average(train, eval_days),
        "baseline_3": cloud_rule(train, eval_days),
    }, index=eval_days.index)
