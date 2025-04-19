import pandas as pd
import numpy as np
import sys
import importlib
import os

# Get the current directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add project root to path
sys.path.append(project_root)

# Import data preprocessing module
from src.utils.data_preprocessing import preprocess_hotel_data

# Define absolute paths
data_file = os.path.join(project_root, 'data', 'raw', 'hotel-booking-demand', 'hotel_bookings.csv')
processed_dir = os.path.join(project_root, 'data', 'processed')

# Load the data
try:
    hotel_df = pd.read_csv(data_file)
    print(f"数据加载成功！形状: {hotel_df.shape}")
    
    # Force reload the module to ensure we have the latest version
    if 'src.utils.data_preprocessing' in sys.modules:
        importlib.reload(sys.modules['src.utils.data_preprocessing'])
    
    # Process the data with error handling
    try:
        processed_df = preprocess_hotel_data(hotel_df)
        print(f"预处理后数据形状: {processed_df.shape}")
        
        # Save the processed data
        os.makedirs(processed_dir, exist_ok=True)
        processed_df.to_csv(os.path.join(processed_dir, 'hotel_processed.csv'), index=False)
        print(f"预处理数据已保存到 {os.path.join(processed_dir, 'hotel_processed.csv')}")
        
        # Continue with feature creation
        from src.utils.data_preprocessing import create_features
        features_df = create_features(processed_df)
        print(f"特征工程后数据形状: {features_df.shape}")
        
        # Save the feature data
        features_df.to_csv(os.path.join(processed_dir, 'hotel_features.csv'), index=False)
        print(f"特征数据已保存到 {os.path.join(processed_dir, 'hotel_features.csv')}")
        
    except ValueError as e:
        print(f"错误: {str(e)}")
        print("尝试使用备用方法...")
        
        # Create a copy for preprocessing
        df_processed = hotel_df.copy()
        
        # Handle date columns manually
        if all(col in df_processed.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
            print("手动创建日期列...")
            
            # Convert month names to numbers
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df_processed['arrival_date_month_num'] = df_processed['arrival_date_month'].map(month_map)
            
            # Create dates with day=1 for all rows to avoid day-out-of-range errors
            df_processed['arrival_date'] = pd.to_datetime(
                df_processed['arrival_date_year'].astype(str) + '-' + 
                df_processed['arrival_date_month_num'].astype(str) + '-01',
                errors='coerce'
            )
            
            # Now safely add the day component where valid
            valid_dates = []
            for _, row in df_processed.iterrows():
                year = row['arrival_date_year']
                month = row['arrival_date_month_num']
                day = row['arrival_date_day_of_month']
                
                # Check if this would be a valid date
                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                    valid_dates.append(date)
                except:
                    # If not valid, use the first of the month
                    valid_dates.append(pd.Timestamp(year=year, month=month, day=1))
            
            df_processed['arrival_date'] = valid_dates
            
            # Extract date features
            df_processed['arrival_dayofweek'] = df_processed['arrival_date'].dt.dayofweek
            df_processed['arrival_quarter'] = df_processed['arrival_date'].dt.quarter
            df_processed['arrival_is_weekend'] = df_processed['arrival_dayofweek'].isin([5, 6]).astype(int)
            
            # Save the processed data
            os.makedirs(processed_dir, exist_ok=True)
            df_processed.to_csv(os.path.join(processed_dir, 'hotel_processed_manual.csv'), index=False)
            print(f"手动预处理数据已保存到 {os.path.join(processed_dir, 'hotel_processed_manual.csv')}")
            
            # Continue with feature creation
            from src.utils.data_preprocessing import create_features
            features_df = create_features(df_processed)
            print(f"特征工程后数据形状: {features_df.shape}")
            
            # Save the feature data
            features_df.to_csv(os.path.join(processed_dir, 'hotel_features_manual.csv'), index=False)
            print(f"特征数据已保存到 {os.path.join(processed_dir, 'hotel_features_manual.csv')}")
        
except FileNotFoundError:
    print(f"数据文件不存在: {data_file}") 