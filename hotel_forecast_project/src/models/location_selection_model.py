import numpy as np
import pandas as pd
import pickle
import os
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("location_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocationSelectionModel:
    """
    酒店选址评分模型
    """
    def __init__(self, model_type='xgboost', params=None):
        """
        初始化选址模型
        
        参数:
        - model_type: 模型类型，可选 'xgboost', 'lightgbm', 'randomforest', 'gbdt'
        - params: 模型参数字典
        """
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        
        # 初始化模型
        self._init_model()
        
        logger.info(f"初始化{model_type}选址模型")
    
    def _init_model(self):
        """
        根据指定类型初始化模型
        """
        if self.model_type == 'xgboost':
            default_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            # 合并默认参数和用户提供的参数
            params = {**default_params, **self.params}
            self.model = xgb.XGBRegressor(**params)
        
        elif self.model_type == 'lightgbm':
            default_params = {
                'objective': 'regression',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            params = {**default_params, **self.params}
            self.model = lgb.LGBMRegressor(**params)
        
        elif self.model_type == 'randomforest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            params = {**default_params, **self.params}
            self.model = RandomForestRegressor(**params)
            
        elif self.model_type == 'gbdt':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            params = {**default_params, **self.params}
            self.model = GradientBoostingRegressor(**params)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def optimize_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='neg_mean_squared_error'):
        """
        通过网格搜索优化超参数
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - param_grid: 参数网格
        - cv: 交叉验证折数
        - scoring: 评分标准
        
        返回:
        - 最佳参数
        """
        logger.info(f"开始{self.model_type}模型超参数优化...")
        
        if param_grid is None:
            # 默认参数网格
            if self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            elif self.model_type == 'lightgbm':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [15, 31, 63],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
            elif self.model_type == 'randomforest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gbdt':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X, y)
        
        # 获取最佳参数
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳分数 ({scoring}): {best_score:.4f}")
        
        # 使用最佳参数更新模型
        self.params.update(best_params)
        self._init_model()
        
        return best_params
    
    def fit(self, X, y, X_val=None, y_val=None, feature_names=None):
        """
        训练模型
        
        参数:
        - X: 特征矩阵
        - y: 目标变量
        - X_val: 验证集特征（可选）
        - y_val: 验证集目标（可选）
        - feature_names: 特征名称列表（可选）
        
        返回:
        - self
        """
        try:
            # 保存特征列名
            self.feature_cols = feature_names if feature_names is not None else \
                                [f"feature_{i}" for i in range(X.shape[1])]
            
            logger.info(f"开始训练{self.model_type}选址模型，特征数量: {len(self.feature_cols)}")
            
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                
                if self.model_type in ['xgboost', 'lightgbm']:
                    eval_set = [(X_scaled, y), (X_val_scaled, y_val)]
                    self.model.fit(X_scaled, y, eval_set=eval_set, 
                                  early_stopping_rounds=10, verbose=False)
                else:
                    self.model.fit(X_scaled, y)
            else:
                self.model.fit(X_scaled, y)
            
            # 计算训练集性能
            y_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            logger.info(f"训练完成. 训练集性能: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            # 如果有验证集，计算验证集性能
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_pred = self.model.predict(X_val_scaled)
                val_mse = mean_squared_error(y_val, y_val_pred)
                val_rmse = np.sqrt(val_mse)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                
                logger.info(f"验证集性能: MSE={val_mse:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}")
            
            return self
        
        except Exception as e:
            logger.error(f"训练模型出错: {str(e)}")
            raise
    
    def predict(self, X):
        """
        预测目标值
        
        参数:
        - X: 特征矩阵
        
        返回:
        - 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()方法训练模型")
        
        # 数据标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        return self.model.predict(X_scaled)
    
    def score_locations(self, locations_df, feature_cols=None, id_col=None):
        """
        为多个候选地址评分
        
        参数:
        - locations_df: 候选地址DataFrame
        - feature_cols: 特征列，如果为None则使用训练时的特征列
        - id_col: 位置ID列名，用于结果
        
        返回:
        - 包含位置得分的DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()方法训练模型")
        
        # 确定特征列
        if feature_cols is None:
            if self.feature_cols is None:
                raise ValueError("未提供特征列且模型未保存特征列信息")
            feature_cols = self.feature_cols
        
        # 提取特征
        X = locations_df[feature_cols].values
        
        # 预测得分
        scores = self.predict(X)
        
        # 创建结果DataFrame
        result_df = locations_df.copy()
        result_df['predicted_score'] = scores
        
        # 排序
        result_df = result_df.sort_values('predicted_score', ascending=False)
        
        return result_df
    
    def get_feature_importance(self, plot=False, n_top=20, figsize=(10, 8)):
        """
        获取特征重要性
        
        参数:
        - plot: 是否绘制特征重要性图
        - n_top: 展示前N个最重要的特征
        - figsize: 图形大小
        
        返回:
        - 特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()方法训练模型")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("模型不支持特征重要性")
            return None
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 显示前n_top个特征
        top_features = feature_importance.head(n_top)
        
        # 绘制特征重要性图
        if plot:
            plt.figure(figsize=figsize)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'特征重要性排名 (Top {n_top})')
            plt.tight_layout()
            plt.show()
        
        return feature_importance
    
    def save_model(self, model_path):
        """
        保存模型到文件
        
        参数:
        - model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit()方法训练模型")
        
        # 创建包含所有必要信息的字典
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_cols': self.feature_cols,
            'scaler': self.scaler
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"模型已保存到: {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """
        从文件加载模型
        
        参数:
        - model_path: 模型文件路径
        
        返回:
        - LocationSelectionModel实例
        """
        try:
            # 加载模型
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 创建类实例
            instance = cls(model_type=model_data['model_type'], params=model_data['params'])
            
            # 加载模型状态
            instance.model = model_data['model']
            instance.feature_cols = model_data['feature_cols']
            instance.scaler = model_data['scaler']
            
            logger.info(f"模型已从{model_path}加载")
            return instance
        
        except Exception as e:
            logger.error(f"加载模型出错: {str(e)}")
            raise


def create_hotel_features(poi_data, distance_data, census_data, competitor_data):
    """
    创建酒店特征数据集
    
    参数:
    - poi_data: POI数据DataFrame
    - distance_data: 距离数据DataFrame
    - census_data: 人口普查数据DataFrame
    - competitor_data: 竞争对手数据DataFrame
    
    返回:
    - 特征DataFrame
    """
    try:
        logger.info("开始创建选址特征...")
        
        # 1. 确保所有数据有相同的位置ID
        location_ids = poi_data['location_id'].unique()
        
        # 2. 合并所有数据源
        features_df = poi_data.set_index('location_id')
        
        # 合并距离数据
        if 'location_id' in distance_data.columns:
            features_df = features_df.join(
                distance_data.set_index('location_id'), how='left')
        
        # 合并人口普查数据
        if 'location_id' in census_data.columns:
            features_df = features_df.join(
                census_data.set_index('location_id'), how='left')
        
        # 合并竞争对手数据
        if 'location_id' in competitor_data.columns:
            features_df = features_df.join(
                competitor_data.set_index('location_id'), how='left')
        
        # 3. 处理缺失值
        features_df = features_df.fillna(0)
        
        # 4. 特征工程
        # 4.1 POI密度特征
        poi_cols = [col for col in features_df.columns if 'poi_' in col]
        if poi_cols:
            features_df['total_poi_count'] = features_df[poi_cols].sum(axis=1)
            
            # 如果有面积信息，计算密度
            if 'area_km2' in features_df.columns:
                features_df['poi_density'] = features_df['total_poi_count'] / features_df['area_km2']
        
        # 4.2 交通便利性得分
        transport_cols = [col for col in features_df.columns if any(x in col for x in ['bus', 'subway', 'train', 'airport'])]
        if transport_cols:
            # 反转距离（越近越好）
            for col in transport_cols:
                if 'distance' in col:
                    features_df[f'{col}_inv'] = 1 / (features_df[col] + 1)  # 加1避免除零
            
            # 交通便利性得分 - 简单求和
            inv_cols = [col for col in features_df.columns if '_inv' in col]
            if inv_cols:
                features_df['transport_score'] = features_df[inv_cols].sum(axis=1)
        
        # 4.3 创建经济指标
        if all(col in features_df.columns for col in ['income_per_capita', 'unemployment_rate']):
            features_df['economic_index'] = features_df['income_per_capita'] * (1 - features_df['unemployment_rate'] / 100)
        
        # 4.4 竞争指数
        if 'competitor_count' in features_df.columns:
            features_df['competition_index'] = features_df['competitor_count'] / (features_df['total_poi_count'] + 1)
        
        # 5. 重置索引
        features_df = features_df.reset_index()
        
        logger.info(f"特征创建完成！特征数量: {features_df.shape[1]}")
        return features_df
    
    except Exception as e:
        logger.error(f"创建特征出错: {str(e)}")
        raise

def evaluate_candidate_locations(model, locations_df, feature_cols, top_n=10):
    """
    评估候选地址，返回排名前N的地址
    
    参数:
    - model: 训练好的LocationSelectionModel
    - locations_df: 候选地址DataFrame
    - feature_cols: 特征列
    - top_n: 返回前N个最佳位置
    
    返回:
    - 包含评分和排名的DataFrame
    """
    # 为位置评分
    scored_locations = model.score_locations(locations_df, feature_cols)
    
    # 添加排名
    scored_locations['rank'] = scored_locations['predicted_score'].rank(ascending=False, method='min')
    
    # 返回前N个
    top_locations = scored_locations.nsmallest(top_n, 'rank')
    
    return top_locations

def analyze_location_factors(model, locations_df, feature_cols, location_id):
    """
    分析影响特定位置评分的因素
    
    参数:
    - model: 训练好的LocationSelectionModel
    - locations_df: 位置数据DataFrame
    - feature_cols: 特征列
    - location_id: 要分析的位置ID
    
    返回:
    - 因素分析DataFrame
    """
    import shap
    
    # 获取特定位置的数据
    location_data = locations_df[locations_df['location_id'] == location_id]
    
    if location_data.empty:
        logger.error(f"未找到位置ID: {location_id}")
        return None
    
    # 获取SHAP值计算器
    explainer = shap.Explainer(model.model)
    
    # 提取特征
    X = location_data[feature_cols].values
    X_scaled = model.scaler.transform(X)
    
    # 计算SHAP值
    shap_values = explainer(X_scaled)
    
    # 创建因素分析DataFrame
    factors_df = pd.DataFrame({
        'feature': feature_cols,
        'value': X[0],
        'impact': shap_values.values[0],
        'abs_impact': np.abs(shap_values.values[0])
    })
    
    # 排序
    factors_df = factors_df.sort_values('abs_impact', ascending=False)
    
    return factors_df 