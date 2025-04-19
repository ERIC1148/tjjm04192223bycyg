import os
from src.utils.data_collection import fetch_poi_data

# 确保目录存在
os.makedirs("data/raw", exist_ok=True)

print("Generating POI data...")

# 定义上海市中心区域的经纬度范围
north = 31.240
south = 31.180
east = 121.540
west = 121.460

# 定义POI类型
tags = {
    "amenity": ["restaurant", "cafe", "hospital", "school", "bank", "hotel", "pharmacy", "parking"],
    "tourism": ["hotel", "attraction", "museum"],
    "shop": ["supermarket", "mall", "convenience"]
}

# 保存POI数据的路径
poi_path = "data/raw/shanghai_poi.geojson"

try:
    # 获取POI数据
    poi_data = fetch_poi_data(north, south, east, west, tags, poi_path)
    print(f"POI data generated and saved to: {poi_path}")
    
    # 打印POI数据统计信息
    if poi_data is not None:
        print(f"Total POIs: {len(poi_data)}")
        
        # 按类型统计POI数量
        poi_types = {}
        for _, row in poi_data.iterrows():
            for key in tags.keys():
                if key in row:
                    poi_type = row[key]
                    if poi_type in poi_types:
                        poi_types[poi_type] += 1
                    else:
                        poi_types[poi_type] = 1
        
        print("\nPOI Count by Type:")
        for poi_type, count in sorted(poi_types.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"- {poi_type}: {count}")
    
except Exception as e:
    print(f"Error generating POI data: {str(e)}")
    
    # 如果获取失败，创建一个简单的模拟POI数据
    print("Creating mock POI data instead...")
    
    import geopandas as gpd
    from shapely.geometry import Point
    import pandas as pd
    import numpy as np
    
    # 创建模拟POI数据
    n_pois = 200
    np.random.seed(42)
    
    # 生成随机POI位置和类型
    geometries = [Point(np.random.uniform(west, east), np.random.uniform(south, north)) for _ in range(n_pois)]
    poi_types = np.random.choice(['restaurant', 'cafe', 'hospital', 'school', 'bank', 'hotel'], n_pois)
    
    # 创建GeoDataFrame
    data = {
        'amenity': poi_types,
        'name': [f"{t}_{i}" for i, t in enumerate(poi_types)],
        'geometry': geometries
    }
    
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # 保存为GeoJSON
    gdf.to_file(poi_path, driver="GeoJSON")
    print(f"Mock POI data created and saved to: {poi_path}")
    print(f"Total POIs: {len(gdf)}")
    
    # 统计POI类型
    type_counts = gdf['amenity'].value_counts()
    print("\nPOI Count by Type:")
    for poi_type, count in type_counts.items():
        print(f"- {poi_type}: {count}")

print("\nPOI data generation completed.") 