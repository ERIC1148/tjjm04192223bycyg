import os
import sys
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"hotel_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. 创建必要的目录
        for directory in ['data/raw', 'data/processed', 'data/models', 'data/results', 'reports', 'logs']:
            os.makedirs(directory, exist_ok=True)
        
        # 2. 生成模拟数据（如果没有真实数据）
        logger.info("Step 1: Generating mock hotel data...")
        if not os.path.exists("data/raw/hotel-booking-demand/hotel_bookings.csv"):
            import create_mock_data
            logger.info("Mock hotel data generated.")
        else:
            logger.info("Hotel data already exists, skipping generation.")
        
        # 3. 生成POI数据
        logger.info("\nStep 2: Generating POI data...")
        if not os.path.exists("data/raw/shanghai_poi.geojson"):
            import generate_poi_data
            logger.info("POI data generated.")
        else:
            logger.info("POI data already exists, skipping generation.")
        
        # 4. 生成天气和假日数据
        logger.info("\nStep 3: Generating weather and holiday data...")
        from src.utils.data_collection import fetch_weather_data, fetch_holiday_data
        
        if not os.path.exists("data/raw/weather_data.csv"):
            weather_df = fetch_weather_data("Shanghai", "2023-01-01", "2023-12-31", "fake_api_key", "data/raw/weather_data.csv")
            logger.info("Weather data generated.")
        else:
            logger.info("Weather data already exists, skipping generation.")
        
        if not os.path.exists("data/raw/holiday_data.csv"):
            holiday_df = fetch_holiday_data("CN", 2023, "data/raw/holiday_data.csv")
            logger.info("Holiday data generated.")
        else:
            logger.info("Holiday data already exists, skipping generation.")
        
        # 5. 预处理数据和训练模型
        logger.info("\nStep 4: Preprocessing data and training models...")
        import train_models
        logger.info("Models trained.")
        
        # 6. 生成预测
        logger.info("\nStep 5: Generating predictions...")
        from src.models.predict import generate_forecasts, evaluate_locations
        
        # 生成测试数据
        if not os.path.exists("data/processed/test_data.csv"):
            logger.info("Creating test data...")
            import pandas as pd
            
            # 使用一部分训练数据作为测试数据
            if os.path.exists("data/processed/hotel_features.csv"):
                df = pd.read_csv("data/processed/hotel_features.csv")
                test_df = df.sample(min(100, len(df)))
                test_df.to_csv("data/processed/test_data.csv", index=False)
                logger.info(f"Test data created from {len(test_df)} samples.")
        
        # 生成候选位置
        if not os.path.exists("data/processed/candidate_locations.csv"):
            logger.info("Creating candidate locations...")
            import pandas as pd
            
            # 使用一部分位置数据作为候选位置
            if os.path.exists("data/processed/location_features.csv"):
                df = pd.read_csv("data/processed/location_features.csv")
                candidate_df = df.sample(min(20, len(df)))
                candidate_df.to_csv("data/processed/candidate_locations.csv", index=False)
                logger.info(f"Candidate locations created from {len(candidate_df)} samples.")
        
        # 生成预测
        if os.path.exists("data/models/spatial_temporal_model.pt") and os.path.exists("data/processed/test_data.csv"):
            logger.info("Generating forecasts...")
            forecasts = generate_forecasts(
                "data/models/spatial_temporal_model.pt", 
                "data/processed/test_data.csv", 
                "data/results/forecasts.csv"
            )
            logger.info("Forecasts generated.")
        
        # 评估候选位置
        if os.path.exists("data/models/location_model.pkl") and os.path.exists("data/processed/candidate_locations.csv"):
            logger.info("Evaluating candidate locations...")
            ranked_locations = evaluate_locations(
                "data/models/location_model.pkl", 
                "data/processed/candidate_locations.csv", 
                "data/results/ranked_locations.csv"
            )
            logger.info("Candidate locations evaluated.")
        
        # 7. 生成报告
        logger.info("\nStep 6: Generating report...")
        if os.path.exists("data/results/forecasts.csv") and os.path.exists("data/results/ranked_locations.csv"):
            from src.utils.reporting import generate_report
            
            report_path = f"reports/hotel_forecast_report_{datetime.now().strftime('%Y%m%d')}.html"
            generate_report("data/results/forecasts.csv", "data/results/ranked_locations.csv", report_path)
            logger.info(f"Report generated at: {report_path}")
        
        logger.info("\nPipeline completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 