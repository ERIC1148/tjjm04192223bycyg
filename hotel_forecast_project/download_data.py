from src.utils.data_collection import download_hotel_booking_data
from src.utils.data_collection import fetch_poi_data
from src.utils.data_collection import fetch_weather_data
from src.utils.data_collection import fetch_holiday_data

# 下载酒店数据
print("Downloading hotel booking data...")
hotel_path = download_hotel_booking_data()
print(f"Hotel data downloaded to: {hotel_path}")

# 获取POI数据
print("\nDownloading POI data...")
north, south, east, west = 31.240, 31.180, 121.540, 121.460
tags = {"amenity": ["restaurant", "hospital", "school", "bank", "hotel"]}
poi_path = "data/raw/shanghai_poi.geojson"
poi_data = fetch_poi_data(north, south, east, west, tags, poi_path)
print(f"POI data downloaded to: {poi_path}")

# 获取天气数据
print("\nGenerating weather data...")
weather_path = "data/raw/weather_data.csv"
weather_data = fetch_weather_data("Shanghai", "2023-01-01", "2023-12-31", "fake_api_key", weather_path)
print(f"Weather data generated at: {weather_path}")

# 获取假期数据
print("\nGenerating holiday data...")
holiday_path = "data/raw/holiday_data.csv"
holiday_data = fetch_holiday_data("CN", 2023, holiday_path)
print(f"Holiday data generated at: {holiday_path}")

print("\nAll data downloaded and generated successfully!") 