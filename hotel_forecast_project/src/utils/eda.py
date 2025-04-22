import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExploratoryDataAnalysis:
    """
    数据探索性分析(EDA)模块 - 基于研究《Exploratory Data Analysis for Feature Selection in Machine Learning》
    提供全面的EDA工具，以提供更好的数据洞察，为模型优化提供数据支持
    """
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col
        
    def generate_summary_stats(self):
        """生成基本统计摘要"""
        summary = {}
        
        # 数值型变量摘要
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        summary['numeric'] = {}
        for col in num_cols:
            summary['numeric'][col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'skew': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'missing': self.df[col].isnull().sum()
            }
        
        # 类别型变量摘要
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        summary['categorical'] = {}
        for col in cat_cols:
            summary['categorical'][col] = {
                'unique_values': self.df[col].nunique(),
                'top_value': self.df[col].value_counts().index[0] if not self.df[col].value_counts().empty else None,
                'top_freq': self.df[col].value_counts().values[0] if not self.df[col].value_counts().empty else 0,
                'missing': self.df[col].isnull().sum()
            }
        
        # 日期型变量摘要
        date_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_cols.append(col)
        
        summary['datetime'] = {}
        for col in date_cols:
            summary['datetime'][col] = {
                'min_date': self.df[col].min(),
                'max_date': self.df[col].max(),
                'range_days': (self.df[col].max() - self.df[col].min()).days,
                'missing': self.df[col].isnull().sum()
            }
        
        return summary
    
    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """
        检测异常值
        
        参数:
        - data: 输入数据，numpy数组或Series
        - method: 检测方法，'iqr'(四分位距法)或'zscore'(Z分数法)
        - threshold: 异常值判定阈值
        
        返回:
        - outlier_mask: 布尔掩码，True表示异常值
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        if method == 'iqr':
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # 创建异常值掩码
            outlier_mask = np.logical_or(
                data < lower_bound,
                data > upper_bound
            )
            
        elif method == 'zscore':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-10))
            outlier_mask = z_scores > threshold
            
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
            
        return outlier_mask
    
    def handle_outliers(self, col_name, method='winsorize', threshold=1.5, return_mask=False):
        """
        处理异常值
        
        参数:
        - col_name: 要处理的列名
        - method: 处理方法，'winsorize'(缩尾)或'remove'(移除)或'log'(对数变换)
        - threshold: IQR方法的阈值乘数
        - return_mask: 是否返回异常值掩码
        
        返回:
        - df_clean: 处理后的DataFrame
        - outliers_mask: (可选) 异常值掩码
        """
        if col_name not in self.df.columns:
            raise ValueError(f"列 {col_name} 不在DataFrame中")
        
        # 检查是否为数值列
        if not np.issubdtype(self.df[col_name].dtype, np.number):
            raise ValueError(f"列 {col_name} 不是数值类型")
        
        # 复制DataFrame避免修改原始数据
        df_clean = self.df.copy()
        
        # 检测异常值
        outlier_mask = self.detect_outliers(df_clean[col_name], method='iqr', threshold=threshold)
        
        # 处理异常值
        if method == 'winsorize':
            # 缩尾法：将异常值替换为阈值
            q1 = np.percentile(df_clean[col_name], 25)
            q3 = np.percentile(df_clean[col_name], 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            df_clean.loc[df_clean[col_name] < lower_bound, col_name] = lower_bound
            df_clean.loc[df_clean[col_name] > upper_bound, col_name] = upper_bound
            
        elif method == 'remove':
            # 移除法：删除异常值对应的样本
            df_clean = df_clean.loc[~outlier_mask].copy()
            
        elif method == 'log':
            # 对数变换：对异常值进行对数变换
            # 先确保所有值为正
            min_val = df_clean[col_name].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                df_clean[col_name] = df_clean[col_name] + shift
            
            # 对异常值应用对数变换
            df_clean.loc[outlier_mask, col_name] = np.log(df_clean.loc[outlier_mask, col_name])
            
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
        
        # 更新数据
        self.df = df_clean
        
        if return_mask:
            return df_clean, outlier_mask
        return df_clean
    
    def plot_distributions(self, cols=None, figsize=(15, 10), output_file=None):
        """绘制数值变量的分布图"""
        if cols is None:
            cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            
        n_cols = min(3, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for i, col in enumerate(cols):
            if i < len(axes):
                ax = axes[i]
                sns.histplot(self.df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
        
        # 隐藏未使用的子图
        for i in range(len(cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"分布图已保存到 {output_file}")
        
        return fig
    
    def plot_target_correlations(self, figsize=(12, 8), output_file=None):
        """绘制与目标变量的相关性图"""
        if self.target_col is None or self.target_col not in self.df.columns:
            raise ValueError("目标变量未指定或不在DataFrame中")
            
        # 选择数值型变量
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        corr_data = self.df[num_cols].corr()[self.target_col].sort_values(ascending=False)
        corr_data = corr_data.drop(self.target_col)  # 移除目标自身
        
        plt.figure(figsize=figsize)
        sns.barplot(x=corr_data.values, y=corr_data.index)
        plt.title(f'特征与目标变量 {self.target_col} 的相关性')
        plt.xlabel('相关系数')
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"目标相关性图已保存到 {output_file}")
        
        return plt.gcf()
    
    def plot_time_series_analysis(self, date_col, value_col, freq='M', figsize=(15, 10), output_file=None):
        """时间序列分析，参考研究《Time Series Analysis for Hotel Demand Forecasting》"""
        if date_col not in self.df.columns:
            raise ValueError(f"日期列 {date_col} 不在DataFrame中")
        if value_col not in self.df.columns:
            raise ValueError(f"值列 {value_col} 不在DataFrame中")
            
        # 确保日期列是日期类型
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
            except:
                raise ValueError(f"无法将列 {date_col} 转换为日期类型")
        
        # 设置日期索引并重采样
        ts_df = self.df.set_index(date_col)
        resampled = ts_df[value_col].resample(freq).mean()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 原始时间序列
        axes[0].plot(resampled)
        axes[0].set_title(f'{value_col} over time')
        axes[0].set_ylabel(value_col)
        
        # 移动平均
        axes[1].plot(resampled.rolling(window=4).mean(), label='4-period MA')
        axes[1].plot(resampled, alpha=0.5, label='Original')
        axes[1].set_title('Moving Average')
        axes[1].legend()
        
        # 季节性分解
        try:
            # 确保数据足够长且没有缺失值
            if len(resampled.dropna()) >= 2 * 12:  # 至少2年的月度数据
                decomposition = seasonal_decompose(resampled.dropna(), model='additive')
                decomposition.seasonal.plot(ax=axes[2])
                axes[2].set_title('Seasonal Component')
            else:
                axes[2].text(0.5, 0.5, 'Insufficient data for seasonal decomposition\n(requires sufficient data)',
                            ha='center', va='center', transform=axes[2].transAxes)
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Seasonal decomposition failed: {str(e)}',
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"时间序列分析图已保存到 {output_file}")
        
        return fig
    
    def plot_spatial_distribution(self, lat_col, lon_col, value_col=None, figsize=(12, 10), output_file=None):
        """空间分布可视化，参考研究《Spatial Distribution Visualization for Hotel Data》"""
        if lat_col not in self.df.columns or lon_col not in self.df.columns:
            raise ValueError(f"经纬度列 {lat_col} 或 {lon_col} 不在DataFrame中")
            
        plt.figure(figsize=figsize)
        
        if value_col is not None and value_col in self.df.columns:
            scatter = plt.scatter(
                self.df[lon_col], 
                self.df[lat_col],
                c=self.df[value_col],
                cmap='viridis',
                alpha=0.7,
                s=50
            )
            plt.colorbar(scatter, label=value_col)
            plt.title(f'Spatial distribution colored by {value_col}')
        else:
            plt.scatter(
                self.df[lon_col],
                self.df[lat_col],
                alpha=0.7,
                s=50
            )
            plt.title('Spatial distribution')
            
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"空间分布图已保存到 {output_file}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_cols, n_top=20, figsize=(12, 8), output_file=None):
        """绘制特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise ValueError("模型必须有feature_importances_属性或get_feature_importance()方法")
            
        indices = np.argsort(importances)[::-1][:n_top]
        
        plt.figure(figsize=figsize)
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存到 {output_file}")
        
        return plt.gcf()
    
    def analyze_and_report(self, output_dir='reports'):
        """生成综合分析报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成摘要统计
        summary = self.generate_summary_stats()
        
        # 保存统计摘要
        summary_path = os.path.join(output_dir, 'summary_stats.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
            
        logger.info(f"统计摘要已保存到 {summary_path}")
        
        # 生成分布图
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        dist_fig = self.plot_distributions(num_cols)
        dist_path = os.path.join(output_dir, 'distributions.png')
        dist_fig.savefig(dist_path, dpi=300, bbox_inches='tight')
        logger.info(f"分布图已保存到 {dist_path}")
        
        # 如果有目标变量，生成相关性图
        if self.target_col is not None:
            try:
                corr_fig = self.plot_target_correlations()
                corr_path = os.path.join(output_dir, 'target_correlations.png')
                corr_fig.savefig(corr_path, dpi=300, bbox_inches='tight')
                logger.info(f"目标相关性图已保存到 {corr_path}")
            except Exception as e:
                logger.warning(f"生成目标相关性图失败: {str(e)}")
        
        # 如果有经纬度列，生成空间分布图
        lat_cols = [col for col in self.df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in self.df.columns if 'lon' in col.lower()]
        if lat_cols and lon_cols:
            try:
                spatial_fig = self.plot_spatial_distribution(lat_cols[0], lon_cols[0], self.target_col)
                spatial_path = os.path.join(output_dir, 'spatial_distribution.png')
                spatial_fig.savefig(spatial_path, dpi=300, bbox_inches='tight')
                logger.info(f"空间分布图已保存到 {spatial_path}")
            except Exception as e:
                logger.warning(f"生成空间分布图失败: {str(e)}")
        
        # 如果有日期列和目标列，生成时间序列分析
        date_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]) or 'date' in col.lower():
                date_cols.append(col)
        
        if date_cols and self.target_col:
            try:
                # 转换日期列
                if not pd.api.types.is_datetime64_any_dtype(self.df[date_cols[0]]):
                    self.df[date_cols[0]] = pd.to_datetime(self.df[date_cols[0]], errors='coerce')
                
                if not self.df[date_cols[0]].isna().all():
                    ts_fig = self.plot_time_series_analysis(date_cols[0], self.target_col)
                    ts_path = os.path.join(output_dir, 'time_series.png')
                    ts_fig.savefig(ts_path, dpi=300, bbox_inches='tight')
                    logger.info(f"时间序列图已保存到 {ts_path}")
            except Exception as e:
                logger.warning(f"时间序列分析出错: {str(e)}")
        
        logger.info(f"分析报告已生成到目录: {output_dir}")
        return os.path.join(output_dir, 'summary_stats.json')

# 实用函数 - 应用于K折交叉验证
def perform_cross_validation(model, X, y, n_splits=5, random_state=42):
    """
    执行K折交叉验证
    
    参数:
    - model: 模型对象，必须有fit和predict方法
    - X: 特征矩阵
    - y: 目标向量
    - n_splits: 折数
    - random_state: 随机种子
    
    返回:
    - 包含交叉验证结果的字典
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = {
        'rmse': [],
        'mae': [],
        'r2': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        cv_scores['rmse'].append(rmse)
        cv_scores['mae'].append(mae)
        cv_scores['r2'].append(r2)
        
        logger.info(f"Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # 计算平均指标
    for metric in cv_scores:
        cv_scores[f'avg_{metric}'] = np.mean(cv_scores[metric])
        cv_scores[f'std_{metric}'] = np.std(cv_scores[metric])
        
    logger.info(f"Average: RMSE={cv_scores['avg_rmse']:.4f}, MAE={cv_scores['avg_mae']:.4f}, R²={cv_scores['avg_r2']:.4f}")
    
    return cv_scores