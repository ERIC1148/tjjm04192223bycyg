import pandas as pd
import logging

logger = logging.getLogger(__name__)

def preprocess_hotel_data(df):
    # ... existing code ...
    if all(col in df_processed.columns for col in ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']):
        logger.info("创建完整到达日期...")
        df_processed['arrival_date'] = pd.to_datetime(
            df_processed['arrival_date_year'].astype(str) + '-' + 
            df_processed['arrival_date_month_num'].astype(str) + '-' + 
            df_processed['arrival_date_day_of_month'].astype(str),
            errors='coerce'  # Convert invalid dates to NaT
        )
        
        # Handle invalid dates (NaT values)
        invalid_dates = df_processed['arrival_date'].isna()
        if invalid_dates.any():
            logger.warning(f"Found {invalid_dates.sum()} invalid dates that were converted to NaT")
            
            # Option 1: Drop rows with invalid dates
            # df_processed = df_processed.dropna(subset=['arrival_date'])
            
            # Option 2: Use a default date for invalid entries (first day of the month)
            df_processed.loc[invalid_dates, 'arrival_date'] = pd.to_datetime(
                df_processed.loc[invalid_dates, 'arrival_date_year'].astype(str) + '-' + 
                df_processed.loc[invalid_dates, 'arrival_date_month_num'].astype(str) + '-01'
            )
        
        # 提取日期特征
        df_processed['arrival_dayofweek'] = df_processed['arrival_date'].dt.dayofweek
    # ... existing code ...
    return df_processed 