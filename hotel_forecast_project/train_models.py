import os
import sys
import pandas as pd
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.utils.data_preprocessing import preprocess_hotel_data, create_features, integrate_poi_data
from src.models.train_models import train_spatial_temporal_model, train_location_model

def main():
    try:
        # 1. 定义路径
        kaggle_path = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub', 'datasets', 
                                  'jessemostipak', 'hotel-booking-demand', 'versions', '1')
        hotel_bookings_csv = os.path.join(kaggle_path, 'hotel_bookings.csv')
        
        # 如果Kaggle数据集不存在，使用生成的模拟数据
        if not os.path.exists(hotel_bookings_csv):
            logger.warning("Kaggle dataset not found, using mock data instead")
            hotel_bookings_csv = "data/raw/hotel-booking-demand/hotel_bookings.csv"
            
            # 检查模拟数据是否存在，如果不存在则运行mock数据脚本
            if not os.path.exists(hotel_bookings_csv):
                logger.info("Generating mock data...")
                import create_mock_data  # 执行生成模拟数据的脚本
        
        # 2. 创建必要的目录
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        
        # 3. 数据预处理
        logger.info("开始数据预处理...")
        hotel_df = pd.read_csv(hotel_bookings_csv)
        logger.info(f"读取酒店数据: {hotel_df.shape}")
        
        # 数据预处理
        processed_path = "data/processed/processed_hotel_data.csv"
        processed_df = preprocess_hotel_data(hotel_df, processed_path)
        logger.info(f"数据预处理完成: {processed_df.shape}")
        
        # 特征工程
        features_path = "data/processed/hotel_features.csv"
        features_df = create_features(processed_df, features_path)
        logger.info(f"特征工程完成: {features_df.shape}")
        
        # 4. 检查是否有POI数据以整合
        poi_path = "data/raw/shanghai_poi.geojson"
        if os.path.exists(poi_path):
            logger.info("整合POI数据...")
            import geopandas as gpd
            poi_gdf = gpd.read_file(poi_path)
            
            integrated_path = "data/processed/hotel_with_poi.csv"
            integrated_df = integrate_poi_data(features_df, poi_gdf, 1.0, integrated_path)
            features_path = integrated_path  # 更新特征路径为整合POI后的路径
            logger.info(f"POI数据整合完成: {integrated_df.shape}")
        
        # 5. 训练空间时间模型
        logger.info("开始训练空间时间模型...")
        st_model_path = "data/models/spatial_temporal_model.pt"
        st_model = train_spatial_temporal_model(features_path, st_model_path)
        logger.info(f"空间时间模型训练完成，保存到: {st_model_path}")
        
        # 6. 训练选址模型
        logger.info("开始训练选址模型...")
        
        # 检查是否存在位置特征数据，如果不存在则创建
        location_data_path = "data/processed/location_features.csv"
        if not os.path.exists(location_data_path):
            logger.info("创建模拟位置数据...")
            import numpy as np
            
            n_locations = 100
            np.random.seed(42)
            
            location_data = {
                'location_id': [f'LOC_{i:03d}' for i in range(n_locations)],
                'latitude': np.random.uniform(31.1, 31.3, n_locations),
                'longitude': np.random.uniform(121.4, 121.6, n_locations),
                'poi_restaurant_count': np.random.randint(5, 50, n_locations),
                'poi_shopping_count': np.random.randint(3, 30, n_locations),
                'poi_entertainment_count': np.random.randint(2, 20, n_locations),
                'poi_transport_count': np.random.randint(1, 15, n_locations),
                'distance_to_city_center': np.random.uniform(0.5, 15.0, n_locations),
                'distance_to_airport': np.random.uniform(5.0, 50.0, n_locations),
                'distance_to_subway': np.random.uniform(0.1, 3.0, n_locations),
                'population_density': np.random.uniform(5000, 25000, n_locations),
                'income_per_capita': np.random.uniform(80000, 150000, n_locations),
                'unemployment_rate': np.random.uniform(2.0, 8.0, n_locations),
                'competitor_count': np.random.randint(0, 10, n_locations),
                'area_km2': np.random.uniform(0.5, 5.0, n_locations),
                'historical_score': np.random.uniform(20, 95, n_locations)
            }
            
            pd.DataFrame(location_data).to_csv(location_data_path, index=False)
        
        location_model_path = "data/models/location_model.pkl"
        location_model = train_location_model(location_data_path, location_model_path)
        logger.info(f"选址模型训练完成，保存到: {location_model_path}")
        
        logger.info("所有模型训练完成!")
        return 0
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 