import json
import os
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
from src.custom_logger import logger

cities = ["Bassens", "Sainte-Eulalie", "Carbon-Blanc", "Yvrac", "Ambarès-et-Lagrave", "Lormont"]

# weather.main -> atm
dict_weather = {
    "clear sky": 1,
    "few clouds": 1,
    "scattered clouds": 8,
    "broken clouds": 8,
    "overcast clouds": 8,
    "shower rain": 2,
    "light rain": 2,
    "moderate rain": 3, 
    "heavy intensity rain": 3,#
    "rain": 3,
    "thunderstorm": 6,
    "light snow": 4,
    "snow": 4,
    "mist": 5,
    "fog": 5, 
    "haze": 5,
}

def to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

def process_weather_data(json_data, target_date_iso=None, time_series=False):
    """
    Process raw weather data from OpenWeather API and extract relevant features.
    
    Args:
        json_data: List of weather forecast data for multiple cities
        target_date_iso: Optional ISO format date string to filter forecasts (e.g., "2026-03-01T12:00:00Z")
        time_series (bool): Whether to process time series data for the next 5 days (True) or just current weather (False).
    
    Returns:
        pd.DataFrame: Processed weather data with atmospheric conditions, daylight, and surface conditions
    """
    
    results = []
    
    # Convert "2026-03-01T12:00:00Z" to a numeric timestamp
    timestamp = None
    if target_date_iso:
        timestamp = datetime.fromisoformat(target_date_iso.replace('Z', '+00:00')).timestamp()

    for city_group in json_data:
        city_name = city_group['city']['name']
        sunrise_second = to_seconds(datetime.fromtimestamp(city_group['city']['sunrise'], tz=timezone.utc))
        sunset_second = to_seconds(datetime.fromtimestamp(city_group['city']['sunset'], tz=timezone.utc))

        if time_series:
            # Process all available forecasts for the city
            forecast_list = city_group['list']
        else:
            # Filter forecasts to keep the one closest to the target timestamp
            forecast_list = city_group['list']
            forecast_list = sorted(forecast_list, key=lambda x: abs(x['dt'] - timestamp))
            forecast_list = forecast_list[:1]  # Keep only the closest forecast
        
        for forecast in forecast_list:
            dt = forecast['dt']
            temp_c = forecast['main']['temp'] - 273.15  # Kelvin -> Celsius conversion
            weather_desc = forecast['weather'][0]['description']
            weather_id = forecast['weather'][0]['id']
            
            # 1. Atmospheric conditions
            atm = dict_weather.get(weather_desc, 9)  # 9 = Other by default
            
            # 2. Light: lighting conditions in which the accident occurred
 
            current_second = to_seconds(datetime.fromtimestamp(dt, tz=timezone.utc))
            if (sunrise_second - 1800) < current_second < (sunset_second + 1800):
                daylight = 1  # Daylight
            elif sunrise_second < current_second < sunset_second:
                daylight = 2  # Twilight or dawn
            else:
                daylight = 0  # Night

            # 3. Surface condition
            if 600 <= weather_id <= 622:
                surf = 5  # Snow-covered
            elif 200 <= weather_id <= 531:
                if temp_c <= 0:
                    surf = 7  # Icy
                else:
                    surf = 2  # Wet
            else:
                surf = 1  # Normal

            results.append({
                "city": city_name,
                "date_time": forecast['dt_txt'],
                "timestamp": dt,
                "atm": atm,
                "daylight": daylight,
                "surf": surf,
                "description": weather_desc,
                "temperature_c": temp_c
            })

    df = pd.DataFrame(results)  
    return df


def get_weather(cities, timestamp=None,time_series=False):
    """
    Fetch weather forecast data for multiple cities from OpenWeather API.
    This function retrieves current weather forecast data for a list of cities
    using the OpenWeather API and processes the results. It handles API errors
    gracefully by logging failures and continuing with other cities.
    Args:
        cities (list): A list of city names to fetch weather data for.
        timestamp (int, optional): Unix timestamp for filtering weather data.
            Defaults to None, which uses the current time for processing.
        time_series (bool): Whether to fetch time series data for the next 5 days (True) or just current weather (False).
    Returns:
        dict or list: Processed weather data returned by process_weather_data(),
            containing weather information for successfully fetched cities.
    Raises:
        KeyError: If OPENWEATHER_API_KEY environment variable is not set.
        requests.RequestException: If network request fails (caught and logged).
    Note:
        - Requires OPENWEATHER_API_KEY environment variable to be set in .env file.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    results = []
    
    for city in cities:
        logger.info(f"Fetching weather data for {city}...")
        
        # Build command
        cmd = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
        response = requests.get(cmd)

        if response.status_code == 200:
            results.append(response.json())
        else:
            logger.error(f"Failed to fetch weather data for {city}. Status code: {response.status_code}")
    logger.info(f"Finished fetching weather data for {len(results)} cities.")
    return process_weather_data(results, timestamp, time_series)


if __name__ == "__main__":
    print(get_weather(cities, "2026-02-11T22:00:00Z"))
