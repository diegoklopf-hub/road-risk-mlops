import numpy as np
import pandas as pd


def encode_date_time(df, timestamp):
    df["year"] = timestamp[:4]
    df["month"] = timestamp[5:7]
    df["day"] = timestamp[8:10]
    df["hrmn"] = timestamp[11:16]

    df["day_of_year"] = pd.to_datetime(df[["year", "month", "day"]]).dt.dayofyear
    df["days_in_year"] = np.where((df["year"].astype(int) % 4 == 0), 366, 365)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / df["days_in_year"])
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / df["days_in_year"])

    # Cyclic encoding of time (hour/minute)
    df["minute"] = (
        pd.to_datetime(df["hrmn"], format="%H:%M").dt.hour * 60
        + pd.to_datetime(df["hrmn"], format="%H:%M").dt.minute
    )
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 1440)

    df["week_day"] = pd.to_datetime(df[["year", "month", "day"]]).dt.weekday
    df["week_day_sin"] = np.sin(2 * np.pi * df["week_day"] / 7)
    df["week_day_cos"] = np.cos(2 * np.pi * df["week_day"] / 7)

    cols_to_drop = ["day_of_year", "days_in_year", "minute", "hrmn", "day", "month", "week_day"]
    df = df.drop(columns=cols_to_drop)
    return df
