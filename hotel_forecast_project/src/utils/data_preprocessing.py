import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import warnings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

def preprocess_hotel_data(df, output_file=None):
    """
    对酒店预订数据进行预处理
    
    参数:
    - df: 输入数据DataFrame
    - output_file: 可选，预处理后数据保存路径
    
    返回:
    - 预处理后的DataFrame
    """
    try:
        logger.info("开始预处理酒店数据...")
        
        # 创建副本，避免修改原始数据
        df_processed = df.copy()
        
        # 1. 处理缺失值
        logger.info("处理缺失值...")
        # 检查缺失值情况
        missing_values = df_processed.isnull().sum()
        logger.info(f"缺失值情况: {missing_values[missing_values > 0].to_dict()}")
        
        # 对于数值型特征，使用中位数填充
        numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                median_value = df_processed[col].median()
                df_processed[col].fillna(median_value, inplace=True)
                logger.info(f"填充缺失值: {col} 使用中位数 {median_value}")
        
        # 对于分类特征，使用众数填充
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().sum() > 0:
                mode_value = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_value, inplace=True)
                logger.info(f"填充缺失值: {col} 使用众数 {mode_value}")
        
        # 2. 处理异常值
        logger.info("处理异常值...")
        # 例如，处理负值或异常大的值
        if 'adr' in df_processed.columns:  # 平均每日房价
            # 替换负值为0
            if (df_processed['adr'] < 0).sum() > 0:
                logger.info(f"发现负的平均每日房价: {(df_processed['adr'] < 0).sum()} 条记录")
                df_processed.loc[df_processed['adr'] < 0, 'adr'] = 0
            
            # 处理异常大的值 (超过99.9%分位数的值)
            q999 = df_processed['adr'].quantile(0.999)
            if (df_processed['adr'] > q999).sum() > 0:
                logger.info(f"发现异常高的平均每日房价 (>{q999}): {(df_processed['adr'] > q999).sum()} 条记录")
                df_processed.loc[df_processed['adr'] > q999, 'adr'] = q999
        
        # 3. 转换分类变量
        logger.info("转换分类变量...")
        # 处理月份 - 转换为数字
        if 'arrival_date_month' in df_processed.columns:
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            df_processed['arrival_date_month_num'] = df_processed['arrival_date_month'].map(month_map)
        
        # 4. 创建日期特征
        logger.info("创建日期特征...")
        # 如果存在年、月、日信息，创建完整日期
        date_columns = [col for col in df_processed.columns if 'date' in col.lower()]
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
                
                # Use the first day of the month for invalid entries
                df_processed.loc[invalid_dates, 'arrival_date'] = pd.to_datetime(
                    df_processed.loc[invalid_dates, 'arrival_date_year'].astype(str) + '-' + 
                    df_processed.loc[invalid_dates, 'arrival_date_month_num'].astype(str) + '-01'
                )
            
            # 提取日期特征
            df_processed['arrival_dayofweek'] = df_processed['arrival_date'].dt.dayofweek
            df_processed['arrival_quarter'] = df_processed['arrival_date'].dt.quarter
            df_processed['arrival_is_weekend'] = df_processed['arrival_dayofweek'].isin([5, 6]).astype(int)
        
        # 5. 创建逗留时间特征
        if all(col in df_processed.columns for col in ['stays_in_weekend_nights', 'stays_in_week_nights']):
            logger.info("创建逗留时间特征...")
            df_processed['total_nights'] = df_processed['stays_in_weekend_nights'] + df_processed['stays_in_week_nights']
            df_processed['weekend_ratio'] = df_processed['stays_in_weekend_nights'] / df_processed['total_nights'].replace(0, 1)
        
        # 6. 创建客人数量特征
        if all(col in df_processed.columns for col in ['adults', 'children', 'babies']):
            logger.info("创建客人数量特征...")
            df_processed['total_guests'] = df_processed['adults'] + df_processed['children'] + df_processed['babies']
            df_processed['has_children'] = ((df_processed['children'] > 0) | (df_processed['babies'] > 0)).astype(int)
        
        # 7. 处理重复项
        duplicate_count = df_processed.duplicated().sum()
        if duplicate_count > 0:
            logger.info(f"删除 {duplicate_count} 条重复记录...")
            df_processed = df_processed.drop_duplicates()
        
        # 8. 保存预处理后的数据
        if output_file:
            logger.info(f"保存预处理后的数据到 {output_file}...")
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_processed.to_csv(output_file, index=False)
        
        logger.info(f"数据预处理完成! 处理前: {df.shape}, 处理后: {df_processed.shape}")
        return df_processed
    
    except Exception as e:
        logger.error(f"数据预处理出错: {str(e)}")
        raise

def create_features(df, output_file=None):
    """
    为预处理后的酒店数据创建特征
    
    参数:
    - df: 预处理后的DataFrame
    - output_file: 可选，特征数据保存路径
    
    返回:
    - 带特征的DataFrame
    """
    try:
        logger.info("开始创建特征...")
        
        # 创建副本，避免修改原始数据
        df_features = df.copy()
        
        # 1. 创建预订提前天数分组
        if 'lead_time' in df_features.columns:
            logger.info("创建预订提前天数分组...")
            df_features['lead_time_group'] = pd.cut(
                df_features['lead_time'],
                bins=[0, 7, 30, 90, 180, 365, float('inf')],
                labels=['last_minute', 'one_week', 'one_month', 'three_months', 'six_months', 'early_bird']
            )
        
        # 2. 创建房价相关特征
        if 'adr' in df_features.columns:
            logger.info("创建房价相关特征...")
            # 按月和酒店类型计算平均房价
            if all(col in df_features.columns for col in ['arrival_date_month', 'hotel']):
                avg_price_by_month_hotel = df_features.groupby(['arrival_date_month', 'hotel'])['adr'].mean().reset_index()
                avg_price_by_month_hotel.rename(columns={'adr': 'avg_price_month_hotel'}, inplace=True)
                
                # 合并回原数据
                df_features = pd.merge(
                    df_features, 
                    avg_price_by_month_hotel,
                    on=['arrival_date_month', 'hotel'],
                    how='left'
                )
                
                # 计算价格比例(个人房价/平均房价)
                df_features['price_ratio'] = df_features['adr'] / df_features['avg_price_month_hotel']
        
        # 3. 客户相关特征
        if 'customer_type' in df_features.columns:
            logger.info("创建客户相关特征...")
            # One-hot编码
            customer_dummies = pd.get_dummies(df_features['customer_type'], prefix='customer')
            df_features = pd.concat([df_features, customer_dummies], axis=1)
        
        # 4. 市场细分特征
        if 'market_segment' in df_features.columns:
            logger.info("创建市场细分特征...")
            # One-hot编码
            market_dummies = pd.get_dummies(df_features['market_segment'], prefix='market')
            df_features = pd.concat([df_features, market_dummies], axis=1)
        
        # 5. 创建是否反复预订特征
        if 'is_repeated_guest' in df_features.columns:
            logger.info("创建反复预订特征...")
            # 如果有预订历史信息，可以计算顾客历史取消率
            if 'reservation_status' in df_features.columns:
                # 这里简化处理，实际项目中应考虑使用更复杂的方法
                df_features['is_returning_customer'] = df_features['is_repeated_guest']
        
        # 6. 季节性特征
        if 'arrival_date_month_num' in df_features.columns:
            logger.info("创建季节性特征...")
            # 创建季节
            month_to_season = {
                1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn',
                11: 'autumn', 12: 'winter'
            }
            df_features['season'] = df_features['arrival_date_month_num'].map(month_to_season)
            
            # 将季节转为one-hot编码
            season_dummies = pd.get_dummies(df_features['season'], prefix='season')
            df_features = pd.concat([df_features, season_dummies], axis=1)
            
            # 创建是否为旺季特征 (假设夏季6-8月和冬季12-2月是旺季)
            peak_months = [1, 2, 6, 7, 8, 12]
            df_features['is_peak_season'] = df_features['arrival_date_month_num'].isin(peak_months).astype(int)
        
        # 7. 交互特征
        logger.info("创建交互特征...")
        # 例如，平均每日房价与住宿时间的交互
        if all(col in df_features.columns for col in ['adr', 'total_nights']):
            df_features['total_price'] = df_features['adr'] * df_features['total_nights']
        
        # 8. 对象类型转为类别类型，提高效率
        for col in df_features.select_dtypes(include=['object']).columns:
            df_features[col] = df_features[col].astype('category')
        
        # 9. 保存特征数据
        if output_file:
            logger.info(f"保存特征数据到 {output_file}...")
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_features.to_csv(output_file, index=False)
        
        logger.info(f"特征创建完成! 输入: {df.shape}, 输出: {df_features.shape}")
        return df_features
    
    except Exception as e:
        logger.error(f"特征创建出错: {str(e)}")
        raise

def integrate_poi_data(hotel_df, poi_gdf, radius_km=1.0, output_file=None):
    """
    将POI数据与酒店数据整合
    
    参数:
    - hotel_df: 酒店数据DataFrame，需包含经纬度坐标
    - poi_gdf: POI数据GeoDataFrame
    - radius_km: 考虑的半径(公里)
    - output_file: 可选，整合后数据保存路径
    
    返回:
    - 整合POI特征后的DataFrame
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from pyproj import CRS
        import numpy as np
        
        logger.info(f"开始整合POI数据，考虑半径: {radius_km}km...")
        
        # 检查酒店数据中是否包含经纬度信息
        if not all(col in hotel_df.columns for col in ['latitude', 'longitude']):
            logger.error("酒店数据缺少经纬度信息!")
            return hotel_df
        
        # 创建酒店位置的几何对象
        geometry = [Point(lon, lat) for lon, lat in zip(hotel_df['longitude'], hotel_df['latitude'])]
        hotel_gdf = gpd.GeoDataFrame(hotel_df, geometry=geometry, crs=CRS.from_epsg(4326))
        
        # 确保POI数据使用相同的坐标参考系统
        if poi_gdf.crs != hotel_gdf.crs:
            poi_gdf = poi_gdf.to_crs(hotel_gdf.crs)
        
        # 创建缓冲区（半径转换为度，近似值）
        # 注意: 1度纬度约等于111公里，1度经度在赤道约等于111公里，但随着纬度增加而减小
        buffer_degrees = radius_km / 111.0
        hotel_gdf['buffer'] = hotel_gdf.geometry.buffer(buffer_degrees)
        
        # 定义要统计的POI类型
        poi_types = []
        if 'amenity' in poi_gdf.columns:
            poi_types = poi_gdf['amenity'].dropna().unique().tolist()
            logger.info(f"发现 {len(poi_types)} 种POI类型: {poi_types[:10]}...")
        
        # 为每种POI类型创建计数特征
        for poi_type in poi_types:
            # 筛选该类型的POI
            type_pois = poi_gdf[poi_gdf['amenity'] == poi_type]
            
            if not type_pois.empty:
                # 统计每个酒店半径内该类型POI的数量
                counts = []
                
                for _, hotel in hotel_gdf.iterrows():
                    # 计算在缓冲区内的POI数量
                    count = sum(type_pois.intersects(hotel['buffer']))
                    counts.append(count)
                
                # 添加计数特征
                column_name = f'poi_{poi_type}_count'
                hotel_gdf[column_name] = counts
                
                logger.info(f"创建POI特征: {column_name}, 均值: {np.mean(counts):.2f}, 最大值: {np.max(counts)}")
        
        # 删除临时缓冲区列
        hotel_gdf = hotel_gdf.drop(columns=['buffer'])
        
        # 转回普通DataFrame
        result_df = pd.DataFrame(hotel_gdf.drop(columns='geometry'))
        
        # 保存整合后的数据
        if output_file:
            logger.info(f"保存整合POI后的数据到 {output_file}...")
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            result_df.to_csv(output_file, index=False)
        
        logger.info(f"POI数据整合完成! 输入: {hotel_df.shape}, 输出: {result_df.shape}")
        return result_df
    
    except Exception as e:
        logger.error(f"整合POI数据出错: {str(e)}")
        # 如果出错，返回原始酒店数据
        return hotel_df


def handle_cold_start(df, hotel_id, hotel_features, output_file=None):
    """
    处理冷启动问题(新酒店没有历史数据)
    
    参数:
    - df: 包含历史数据的DataFrame
    - hotel_id: 新酒店ID
    - hotel_features: 新酒店特征字典
    - output_file: 可选，处理后数据保存路径
    
    返回:
    - 处理后的DataFrame，包含基于相似酒店的预测
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        logger.info(f"开始处理冷启动问题，酒店ID: {hotel_id}...")
        
        # 提取酒店静态特征
        static_features = [
            'hotel', 'city', 'poi_restaurant_count', 'poi_park_count', 
            'poi_shopping_count', 'distance_to_city_center'
        ]
        
        # 过滤出存在的特征
        existing_features = [f for f in static_features if f in df.columns]
        
        if not existing_features:
            logger.error("找不到用于冷启动的特征!")
            return df
        
        logger.info(f"使用特征: {existing_features}")
        
        # 提取现有酒店的特征
        X = df[existing_features].drop_duplicates(subset=['hotel'])
        
        # 创建新酒店的特征向量
        new_hotel_features = {k: v for k, v in hotel_features.items() if k in existing_features}
        new_hotel_df = pd.DataFrame([new_hotel_features])
        
        # 对分类特征进行编码
        cat_features = [f for f in existing_features if df[f].dtype == 'object' or df[f].dtype.name == 'category']
        num_features = [f for f in existing_features if f not in cat_features]
        
        # 处理分类特征
        for feature in cat_features:
            # 对现有酒店和新酒店使用相同的编码
            all_values = pd.concat([X[feature], new_hotel_df[feature]]).unique()
            value_to_idx = {val: i for i, val in enumerate(all_values)}
            
            X[f"{feature}_code"] = X[feature].map(value_to_idx)
            new_hotel_df[f"{feature}_code"] = new_hotel_df[feature].map(value_to_idx)
            
            # 更新特征列表
            existing_features.remove(feature)
            existing_features.append(f"{feature}_code")
        
        # 确保列表中只包含数值特征
        existing_features = [f for f in existing_features if f in X.columns and X[f].dtype in ['int64', 'float64']]
        
        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[existing_features])
        new_hotel_scaled = scaler.transform(new_hotel_df[existing_features])
        
        # 找到最相似的K个酒店
        k = min(5, len(X))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, indices = nbrs.kneighbors(new_hotel_scaled)
        
        # 获取相似酒店的ID
        similar_hotels = X.iloc[indices[0]]['hotel'].tolist()
        logger.info(f"找到{k}个最相似的酒店: {similar_hotels}")
        
        # 计算新酒店的预测数据 (使用相似酒店的平均值)
        similar_data = df[df['hotel'].isin(similar_hotels)]
        
        # 按照预订日期和客人类型等分组，计算平均值
        if all(col in df.columns for col in ['arrival_date_month', 'customer_type']):
            group_cols = ['arrival_date_month', 'customer_type']
            
            # 计算分组平均值
            aggregations = {}
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if col != 'hotel_id' and col not in group_cols:
                    aggregations[col] = 'mean'
            
            predictions = similar_data.groupby(group_cols).agg(aggregations).reset_index()
            
            # 添加新酒店ID
            predictions['hotel'] = hotel_id
            
            # 标记为预测数据
            predictions['is_prediction'] = 1
            
            # 将预测添加到原始数据中
            result_df = pd.concat([df, predictions], ignore_index=True)
        else:
            logger.warning("缺少必要的分组列，使用简单平均")
            predictions = similar_data.mean(numeric_only=True).to_dict()
            predictions['hotel'] = hotel_id
            predictions['is_prediction'] = 1
            
            # 添加为新行
            result_df = df.copy()
            result_df = result_df.append(predictions, ignore_index=True)
        
        # 保存处理后的数据
        if output_file:
            logger.info(f"保存冷启动处理后的数据到 {output_file}...")
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            result_df.to_csv(output_file, index=False)
        
        logger.info(f"冷启动处理完成! 输入: {df.shape}, 输出: {result_df.shape}")
        return result_df
    
    except Exception as e:
        logger.error(f"冷启动处理出错: {str(e)}")
        # 如果出错，返回原始数据
        return df 