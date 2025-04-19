import os
import kagglehub
import osmnx as ox
import geopandas as gpd
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_hotel_booking_data():
    """
    从Kaggle下载酒店预订数据集
    """
    try:
        logger.info("开始下载酒店预订数据集...")
        path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
        logger.info(f"酒店预订数据集已下载到: {path}")
        return path
    except Exception as e:
        logger.error(f"下载酒店预订数据集时出错: {str(e)}")
        raise

def fetch_poi_data(north, south, east, west, tags, output_file="data/raw/poi_data.geojson"):
    """
    从OpenStreetMap获取指定区域的POI数据
    
    参数:
    - north, south, east, west: 经纬度边界框
    - tags: POI类型标签, 如 {"amenity": ["restaurant", "hotel"]}
    - output_file: 输出文件路径
    
    返回:
    - GeoDataFrame对象
    """
    try:
        logger.info(f"获取区域 ({north}, {south}, {east}, {west}) 的POI数据...")
        gdf = ox.features_from_bbox(north, south, east, west, tags)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为GeoJSON
        gdf.to_file(output_file, driver="GeoJSON")
        logger.info(f"POI数据已保存到: {output_file}")
        return gdf
    except Exception as e:
        logger.error(f"获取POI数据时出错: {str(e)}")
        raise

def fetch_weather_data(city, start_date, end_date, api_key, output_file="data/raw/weather_data.csv"):
    """
    从OpenWeatherMap获取历史天气数据
    
    参数:
    - city: 城市名称
    - start_date, end_date: 开始和结束日期 (YYYY-MM-DD格式)
    - api_key: OpenWeatherMap API密钥
    - output_file: 输出文件路径
    
    返回:
    - DataFrame对象
    """
    try:
        logger.info(f"获取{city}从{start_date}到{end_date}的天气数据...")
        
        # 将日期字符串转换为datetime对象
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 构建日期列表
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        # 示例API调用（实际项目中需使用真实API）
        weather_data = []
        for date in dates:
            # 这里应该是实际的API调用，这里仅为示例
            # url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&dt={int(datetime.strptime(date, '%Y-%m-%d').timestamp())}&appid={api_key}"
            # response = requests.get(url)
            # data = response.json()
            
            # 为演示创建模拟数据
            weather_data.append({
                "date": date,
                "temp_max": 25 + (datetime.strptime(date, "%Y-%m-%d").day % 10),
                "temp_min": 15 + (datetime.strptime(date, "%Y-%m-%d").day % 8),
                "humidity": 50 + (datetime.strptime(date, "%Y-%m-%d").day % 30),
                "precipitation": (datetime.strptime(date, "%Y-%m-%d").day % 10) / 10,
            })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(weather_data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"天气数据已保存到: {output_file}")
        return df
    except Exception as e:
        logger.error(f"获取天气数据时出错: {str(e)}")
        raise

def fetch_holiday_data(country, year, output_file="data/raw/holiday_data.csv"):
    """
    获取指定国家和年份的假期数据
    
    参数:
    - country: 国家代码 (如 'CN' 表示中国)
    - year: 年份
    - output_file: 输出文件路径
    
    返回:
    - DataFrame对象
    """
    try:
        logger.info(f"获取{country} {year}年的假期数据...")
        
        # 实际项目中应从公共API或GitHub repo获取假期数据
        # 这里为演示创建模拟数据
        
        # 中国主要假期（简化版）
        cn_holidays = [
            {"date": f"{year}-01-01", "name": "元旦", "is_holiday": True},
            {"date": f"{year}-01-02", "name": "元旦假期", "is_holiday": True},
            {"date": f"{year}-01-03", "name": "元旦假期", "is_holiday": True},
            {"date": f"{year}-02-10", "name": "春节", "is_holiday": True},
            {"date": f"{year}-02-11", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-02-12", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-02-13", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-02-14", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-02-15", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-02-16", "name": "春节假期", "is_holiday": True},
            {"date": f"{year}-04-05", "name": "清明节", "is_holiday": True},
            {"date": f"{year}-05-01", "name": "劳动节", "is_holiday": True},
            {"date": f"{year}-05-02", "name": "劳动节假期", "is_holiday": True},
            {"date": f"{year}-05-03", "name": "劳动节假期", "is_holiday": True},
            {"date": f"{year}-05-04", "name": "劳动节假期", "is_holiday": True},
            {"date": f"{year}-05-05", "name": "劳动节假期", "is_holiday": True},
            {"date": f"{year}-06-25", "name": "端午节", "is_holiday": True},
            {"date": f"{year}-09-19", "name": "中秋节", "is_holiday": True},
            {"date": f"{year}-10-01", "name": "国庆节", "is_holiday": True},
            {"date": f"{year}-10-02", "name": "国庆节假期", "is_holiday": True},
            {"date": f"{year}-10-03", "name": "国庆节假期", "is_holiday": True},
            {"date": f"{year}-10-04", "name": "国庆节假期", "is_holiday": True},
            {"date": f"{year}-10-05", "name": "国庆节假期", "is_holiday": True},
            {"date": f"{year}-10-06", "name": "国庆节假期", "is_holiday": True},
            {"date": f"{year}-10-07", "name": "国庆节假期", "is_holiday": True},
        ]
        
        holiday_data = cn_holidays if country == 'CN' else []
        
        # 创建DataFrame并保存
        df = pd.DataFrame(holiday_data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"假期数据已保存到: {output_file}")
        return df
    except Exception as e:
        logger.error(f"获取假期数据时出错: {str(e)}")
        raise

def fetch_traffic_data(city, date, api_key, output_file="data/raw/traffic_data.csv"):
    """
    获取交通流量数据（模拟，实际项目中应使用百度地图或高德地图API）
    
    参数:
    - city: 城市名称
    - date: 日期
    - api_key: API密钥
    - output_file: 输出文件路径
    
    返回:
    - DataFrame对象
    """
    try:
        logger.info(f"获取{city} {date}的交通流量数据...")
        
        # 模拟一天24小时的交通流量数据
        hours = list(range(24))
        traffic_levels = []
        
        # 生成模拟数据 - 早晚高峰
        for hour in hours:
            if 7 <= hour <= 9:  # 早高峰
                level = 80 + (hour - 7) * 10
            elif 17 <= hour <= 19:  # 晚高峰
                level = 85 + (hour - 17) * 5
            elif 23 <= hour or hour <= 5:  # 深夜/凌晨
                level = 20 + hour * 2 if hour <= 5 else 20 + (24 - hour) * 2
            else:  # 其他时间
                level = 40 + hour
            
            # 添加一些随机变化
            import random
            level = min(100, max(0, level + random.randint(-10, 10)))
            
            traffic_levels.append(level)
        
        # 创建DataFrame
        traffic_data = {
            "hour": hours,
            "traffic_level": traffic_levels,
            "date": date,
            "city": city
        }
        
        df = pd.DataFrame(traffic_data)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"交通流量数据已保存到: {output_file}")
        return df
    except Exception as e:
        logger.error(f"获取交通流量数据时出错: {str(e)}")
        raise