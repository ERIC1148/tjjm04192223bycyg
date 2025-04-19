import os
import sys
import json
import logging
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from src.utils.data_collection import download_hotel_booking_data, fetch_poi_data, fetch_weather_data, fetch_holiday_data
from src.utils.data_preprocessing import preprocess_hotel_data, create_features, integrate_poi_data
from src.models.train_models import train_spatial_temporal_model, train_location_model
from src.models.predict import generate_forecasts, evaluate_locations
from src.utils.reporting import generate_report

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "rpa_automation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保日志目录存在
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

class RPAAutomation:
    """
    RPA自动化接口类，用于与UiPath进行交互
    """
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.project_root = project_root
        logger.info(f"RPA自动化初始化完成，项目根目录: {self.project_root}")
    
    def _load_config(self, config_path=None):
        """加载配置文件"""
        default_config = {
            "data_dir": os.path.join(project_root, "data"),
            "raw_data_dir": os.path.join(project_root, "data", "raw"),
            "processed_data_dir": os.path.join(project_root, "data", "processed"),
            "models_dir": os.path.join(project_root, "data", "models"),
            "results_dir": os.path.join(project_root, "data", "results"),
            "reports_dir": os.path.join(project_root, "reports"),
            "api_keys": {
                "weather": "",
                "traffic": ""
            },
            "email": {
                "recipients": ["stakeholders@example.com"],
                "subject": "酒店选址与客流预测报告 - {date}"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并用户配置与默认配置
                    for key, value in config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"从 {config_path} 加载配置")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        # 确保所有目录存在
        for key, path in default_config.items():
            if key.endswith('_dir') and isinstance(path, str):
                os.makedirs(path, exist_ok=True)
        
        return default_config
    
    def download_data(self):
        """下载数据集"""
        try:
            logger.info("开始下载数据...")
            
            # 下载酒店预订数据
            kaggle_path = download_hotel_booking_data(self.config["raw_data_dir"])
            logger.info(f"酒店预订数据下载完成: {kaggle_path}")
            
            # 获取POI数据
            north, south, east, west = 31.240, 31.180, 121.540, 121.460
            tags = {"amenity": ["restaurant", "hospital", "school", "bank", "hotel"]}
            poi_path = os.path.join(self.config["raw_data_dir"], "shanghai_poi.geojson")
            
            gdf = fetch_poi_data(north, south, east, west, tags, poi_path)
            logger.info(f"POI数据获取完成: {poi_path}")
            
            # 获取天气数据
            weather_path = os.path.join(self.config["raw_data_dir"], "weather_data.csv")
            weather_df = fetch_weather_data("Shanghai", "2023-01-01", "2023-12-31", 
                                          self.config["api_keys"]["weather"], weather_path)
            logger.info(f"天气数据获取完成: {weather_path}")
            
            # 获取假期数据
            holiday_path = os.path.join(self.config["raw_data_dir"], "holiday_data.csv")
            holiday_df = fetch_holiday_data("CN", 2023, holiday_path)
            logger.info(f"假期数据获取完成: {holiday_path}")
            
            return {
                "status": "success",
                "data": {
                    "hotel_data": kaggle_path,
                    "poi_data": poi_path,
                    "weather_data": weather_path,
                    "holiday_data": holiday_path
                }
            }
        except Exception as e:
            logger.error(f"下载数据失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def preprocess_data(self):
        """预处理数据"""
        try:
            logger.info("开始预处理数据...")
            
            # 读取酒店预订数据
            hotel_df_path = os.path.join(self.config["raw_data_dir"], "hotel-booking-demand", "hotel_bookings.csv")
            if not os.path.exists(hotel_df_path):
                logger.error(f"酒店预订数据文件不存在: {hotel_df_path}")
                raise FileNotFoundError(f"找不到文件: {hotel_df_path}")
            
            hotel_df = pd.read_csv(hotel_df_path)
            logger.info(f"加载酒店数据成功，形状: {hotel_df.shape}")
            
            # 预处理酒店数据
            processed_path = os.path.join(self.config["processed_data_dir"], "processed_hotel_data.csv")
            processed_df = preprocess_hotel_data(hotel_df, processed_path)
            logger.info(f"酒店数据预处理完成: {processed_path}")
            
            # 特征工程
            features_path = os.path.join(self.config["processed_data_dir"], "hotel_features.csv")
            features_df = create_features(processed_df, features_path)
            logger.info(f"特征工程完成: {features_path}")
            
            # 整合POI数据
            poi_path = os.path.join(self.config["raw_data_dir"], "shanghai_poi.geojson")
            if os.path.exists(poi_path):
                import geopandas as gpd
                poi_gdf = gpd.read_file(poi_path)
                
                integrated_path = os.path.join(self.config["processed_data_dir"], "hotel_with_poi.csv")
                integrated_df = integrate_poi_data(features_df, poi_gdf, 1.0, integrated_path)
                logger.info(f"POI数据整合完成: {integrated_path}")
            else:
                logger.warning(f"POI数据文件不存在: {poi_path}，跳过POI整合")
                integrated_path = features_path
            
            return {
                "status": "success",
                "data": {
                    "processed_data": processed_path,
                    "features_data": features_path,
                    "integrated_data": integrated_path
                }
            }
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def train_models(self):
        """训练模型"""
        try:
            logger.info("开始训练模型...")
            
            # 训练空间时间模型
            features_path = os.path.join(self.config["processed_data_dir"], "hotel_with_poi.csv")
            if not os.path.exists(features_path):
                features_path = os.path.join(self.config["processed_data_dir"], "hotel_features.csv")
            
            if not os.path.exists(features_path):
                logger.error(f"特征数据文件不存在: {features_path}")
                raise FileNotFoundError(f"找不到文件: {features_path}")
            
            st_model_path = os.path.join(self.config["models_dir"], "spatial_temporal_model.pt")
            st_model = train_spatial_temporal_model(features_path, st_model_path)
            logger.info(f"空间时间模型训练完成: {st_model_path}")
            
            # 训练选址模型
            # 为了演示，创建模拟位置数据
            location_data_path = os.path.join(self.config["processed_data_dir"], "location_features.csv")
            if not os.path.exists(location_data_path):
                logger.warning(f"位置数据不存在: {location_data_path}，生成模拟数据")
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
            
            location_model_path = os.path.join(self.config["models_dir"], "location_model.pkl")
            location_model = train_location_model(location_data_path, location_model_path)
            logger.info(f"选址模型训练完成: {location_model_path}")
            
            return {
                "status": "success",
                "data": {
                    "spatial_temporal_model": st_model_path,
                    "location_model": location_model_path
                }
            }
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def generate_predictions(self):
        """生成预测结果"""
        try:
            logger.info("开始生成预测...")
            
            # 加载测试数据或创建模拟数据
            test_data_path = os.path.join(self.config["processed_data_dir"], "test_data.csv")
            if not os.path.exists(test_data_path):
                logger.warning(f"测试数据不存在: {test_data_path}，使用模拟数据")
                # 这里简化处理，实际应有更复杂的测试数据生成逻辑
                features_path = os.path.join(self.config["processed_data_dir"], "hotel_features.csv")
                if os.path.exists(features_path):
                    df = pd.read_csv(features_path)
                    if len(df) > 100:
                        test_df = df.sample(100)
                    else:
                        test_df = df
                    test_df.to_csv(test_data_path, index=False)
                else:
                    logger.error(f"找不到特征数据来生成测试数据: {features_path}")
                    raise FileNotFoundError(f"找不到文件: {features_path}")
            
            # 加载候选位置数据或创建模拟数据
            candidate_path = os.path.join(self.config["processed_data_dir"], "candidate_locations.csv")
            if not os.path.exists(candidate_path):
                logger.warning(f"候选位置数据不存在: {candidate_path}，使用模拟数据")
                location_data_path = os.path.join(self.config["processed_data_dir"], "location_features.csv")
                if os.path.exists(location_data_path):
                    df = pd.read_csv(location_data_path)
                    if len(df) > 20:
                        candidate_df = df.sample(20)
                    else:
                        candidate_df = df
                    candidate_df.to_csv(candidate_path, index=False)
                else:
                    logger.error(f"找不到位置数据来生成候选位置: {location_data_path}")
                    raise FileNotFoundError(f"找不到文件: {location_data_path}")
            
            # 加载模型并生成预测
            st_model_path = os.path.join(self.config["models_dir"], "spatial_temporal_model.pt")
            location_model_path = os.path.join(self.config["models_dir"], "location_model.pkl")
            
            # 生成客流预测
            forecasts_path = os.path.join(self.config["results_dir"], "forecasts.csv")
            forecasts = generate_forecasts(st_model_path, test_data_path, forecasts_path)
            logger.info(f"客流预测完成: {forecasts_path}")
            
            # 评估候选位置
            ranked_path = os.path.join(self.config["results_dir"], "ranked_locations.csv")
            ranked_locations = evaluate_locations(location_model_path, candidate_path, ranked_path)
            logger.info(f"位置评估完成: {ranked_path}")
            
            return {
                "status": "success",
                "data": {
                    "forecasts": forecasts_path,
                    "ranked_locations": ranked_path
                }
            }
        except Exception as e:
            logger.error(f"生成预测失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def generate_report(self):
        """生成报告"""
        try:
            logger.info("开始生成报告...")
            
            # 加载预测结果
            forecasts_path = os.path.join(self.config["results_dir"], "forecasts.csv")
            ranked_path = os.path.join(self.config["results_dir"], "ranked_locations.csv")
            
            if not os.path.exists(forecasts_path):
                logger.error(f"客流预测结果不存在: {forecasts_path}")
                raise FileNotFoundError(f"找不到文件: {forecasts_path}")
            
            if not os.path.exists(ranked_path):
                logger.error(f"位置评估结果不存在: {ranked_path}")
                raise FileNotFoundError(f"找不到文件: {ranked_path}")
            
            # 生成报告
            report_path = os.path.join(self.config["reports_dir"], f"hotel_forecast_report_{datetime.now().strftime('%Y%m%d')}.html")
            generate_report(forecasts_path, ranked_path, report_path)
            logger.info(f"报告生成完成: {report_path}")
            
            return {
                "status": "success",
                "data": {
                    "report": report_path
                }
            }
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def run_full_pipeline(self):
        """运行完整流程"""
        logger.info("开始运行完整流程...")
        
        results = {
            "download_data": self.download_data(),
            "preprocess_data": self.preprocess_data(),
            "train_models": self.train_models(),
            "generate_predictions": self.generate_predictions(),
            "generate_report": self.generate_report()
        }
        
        success = all(step["status"] == "success" for step in results.values())
        
        if success:
            logger.info("完整流程运行成功!")
        else:
            failed_steps = [step for step, result in results.items() if result["status"] != "success"]
            logger.error(f"流程运行失败! 失败步骤: {failed_steps}")
        
        return {
            "status": "success" if success else "error",
            "results": results
        }

def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='酒店选址与客流预测RPA自动化')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--step', type=str, choices=['download', 'preprocess', 'train', 'predict', 'report', 'all'], 
                        default='all', help='执行的步骤')
    
    args = parser.parse_args()
    
    # 初始化RPA自动化
    rpa = RPAAutomation(args.config)
    
    # 根据参数执行相应步骤
    if args.step == 'download':
        result = rpa.download_data()
    elif args.step == 'preprocess':
        result = rpa.preprocess_data()
    elif args.step == 'train':
        result = rpa.train_models()
    elif args.step == 'predict':
        result = rpa.generate_predictions()
    elif args.step == 'report':
        result = rpa.generate_report()
    else:  # 'all'
        result = rpa.run_full_pipeline()
    
    # 输出结果
    print(json.dumps(result, indent=2))
    
    return 0 if result["status"] == "success" else 1

if __name__ == "__main__":
    sys.exit(main())