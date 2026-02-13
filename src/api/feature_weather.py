import pandas as pd

from weather import get_weather
from feature_time import encode_date_time


def encode_meteorological_features(df, cities, timestamp, time_series, secteur, n=10):
    weather_df = get_weather(cities=cities, timestamp=timestamp, time_series=time_series)
    nb_range_max = weather_df.shape[0] // len(cities)
    n_to_process = min(n, nb_range_max)

    results = []

    for idx in range(n_to_process):
        df_it = df.copy()  # Copy the road set for this time slot

        # Initialize with the first city's first-slot timestamp so all rows share the same time key.
        current_ts = weather_df[weather_df["city"] == cities[0]].iloc[idx]["timestamp"]
        df_it["prediction_time"] = current_ts

        # Encode time features
        slot_dt = pd.to_datetime(current_ts, unit="s")
        slot_ts_iso = slot_dt.strftime("%Y-%m-%dT%H:%M")
        df_it = encode_date_time(df_it, slot_ts_iso)

        for city in cities:
            city_weather = weather_df[weather_df["city"] == city].iloc[idx]
            insee_code = secteur[city]
            mask_city = df_it["com"] == insee_code

            # Update weather features
            df_it.loc[mask_city, "atm"] = city_weather["atm"]
            df_it.loc[mask_city, "surf"] = city_weather["surf"]
            df_it.loc[mask_city, "temperature_c"] = city_weather["temperature_c"]
            df_it.loc[mask_city, "description"] = city_weather["description"]
            df_it.loc[mask_city, "daylight"] = city_weather["daylight"]

            # Luminosity logic (night/day)
            is_night = city_weather["daylight"] == 0
            if is_night:
                df_it.loc[mask_city & (df_it["agg"] == 1), "lum"] = 3
                df_it.loc[mask_city & (df_it["agg"] == 2), "lum"] = 5
            else:
                df_it.loc[mask_city, "lum"] = city_weather["daylight"]

        results.append(df_it)

    df_meteo_timestamp = pd.concat(results, ignore_index=True)

    return df_meteo_timestamp
