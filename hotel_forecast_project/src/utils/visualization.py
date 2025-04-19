import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置默认样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def plot_feature_correlation(df, feature_cols=None, figsize=(12, 10), output_file=None):
    """
    绘制特征相关性热图
    
    参数:
    - df: 数据DataFrame
    - feature_cols: 要包含的特征列列表，如果为None则使用所有数值列
    - figsize: 图形大小
    - output_file: 输出文件路径，如果为None则不保存
    
    返回:
    - fig, ax: matplotlib图形对象
    """
    try:
        logger.info("绘制特征相关性热图...")
        
        # 如果未指定特征列，则使用所有数值列
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # 计算相关性矩阵
        corr_matrix = df[feature_cols].corr()
        
        # 创建图形和轴对象
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)
        
        plt.title('特征相关性矩阵', fontsize=16)
        plt.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"相关性热图已保存到 {output_file}")
        
        return fig, ax
    except Exception as e:
        logger.error(f"绘制特征相关性热图时出错: {str(e)}")
        raise

def plot_time_series(df, date_col, value_col, category_col=None, title=None, 
                     figsize=(12, 6), output_file=None):
    """
    绘制时间序列数据
    
    参数:
    - df: 数据DataFrame
    - date_col: 日期列名
    - value_col: 值列名
    - category_col: 分类列名，如果提供则按类别绘制多条线
    - title: 图表标题
    - figsize: 图形大小
    - output_file: 输出文件路径，如果为None则不保存
    
    返回:
    - fig, ax: matplotlib图形对象
    """
    try:
        logger.info("绘制时间序列图...")
        
        # 确保日期列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
        
        # 创建图形和轴对象
        fig, ax = plt.subplots(figsize=figsize)
        
        # 如果有分类列，按类别绘制多条线
        if category_col:
            for category, group in df.groupby(category_col):
                group = group.sort_values(date_col)
                ax.plot(group[date_col], group[value_col], marker='o', linestyle='-', 
                        label=f'{category_col}={category}')
            ax.legend()
        else:
            # 按日期排序
            df_sorted = df.sort_values(date_col)
            ax.plot(df_sorted[date_col], df_sorted[value_col], marker='o', linestyle='-')
        
        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        
        # 格式化x轴日期标签
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"时间序列图已保存到 {output_file}")
        
        return fig, ax
    except Exception as e:
        logger.error(f"绘制时间序列图时出错: {str(e)}")
        raise

def plot_actual_vs_predicted(y_true, y_pred, title=None, figsize=(10, 8), output_file=None):
    """
    绘制实际值与预测值的散点图
    
    参数:
    - y_true: 实际值数组或Series
    - y_pred: 预测值数组或Series
    - title: 图表标题
    - figsize: 图形大小
    - output_file: 输出文件路径，如果为None则不保存
    
    返回:
    - fig, ax: matplotlib图形对象
    """
    try:
        logger.info("绘制实际值与预测值对比图...")
        
        # 创建图形和轴对象
        fig, ax = plt.subplots(figsize=figsize)
        
        # 散点图
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # 添加对角线 (理想预测线)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # 设置标题和标签
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title('实际值 vs 预测值', fontsize=14)
        
        ax.set_xlabel('实际值', fontsize=12)
        ax.set_ylabel('预测值', fontsize=12)
        
        # 计算R² 并添加文本标注
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        fig.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"实际值与预测值对比图已保存到 {output_file}")
        
        return fig, ax
    except Exception as e:
        logger.error(f"绘制实际值与预测值对比图时出错: {str(e)}")
        raise

def plot_feature_importance(model, feature_names, n_top=20, figsize=(12, 10), output_file=None):
    """
    绘制特征重要性条形图
    
    参数:
    - model: 模型对象，必须有feature_importances_属性或能通过get_feature_importance()方法获取特征重要性
    - feature_names: 特征名称列表
    - n_top: 显示前n个最重要的特征
    - figsize: 图形大小
    - output_file: 输出文件路径，如果为None则不保存
    
    返回:
    - fig, ax: matplotlib图形对象
    """
    try:
        logger.info("绘制特征重要性图...")
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise AttributeError("模型没有feature_importances_属性或get_feature_importance()方法")
        
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
        
        # 按重要性降序排序
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # 取前n_top个特征
        if n_top < len(feature_importance):
            feature_importance = feature_importance.iloc[:n_top]
        
        # 创建图形和轴对象
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制条形图
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        
        # 设置标题和标签
        ax.set_title('特征重要性', fontsize=14)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        
        fig.tight_layout()
        
        # 保存图形
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图已保存到 {output_file}")
        
        return fig, ax
    except Exception as e:
        logger.error(f"绘制特征重要性图时出错: {str(e)}")
        raise

def plot_map_with_locations(df, lat_col='latitude', lon_col='longitude', value_col=None, 
                           title=None, zoom_start=12, output_file=None):
    """
    使用folium在地图上绘制位置点
    
    参数:
    - df: 包含位置数据的DataFrame
    - lat_col: 纬度列名
    - lon_col: 经度列名
    - value_col: 值列名，用于标记点的颜色和大小
    - title: 地图标题
    - zoom_start: 初始缩放级别
    - output_file: 输出HTML文件路径，如果为None则不保存
    
    返回:
    - folium.Map对象
    """
    try:
        logger.info("生成位置地图...")
        
        # 计算地图中心点
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        
        # 创建地图
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start,
                      tiles='CartoDB positron')
        
        # 如果提供了标题，添加标题
        if title:
            title_html = f'<h3 align="center" style="font-size:16px"><b>{title}</b></h3>'
            m.get_root().html.add_child(folium.Element(title_html))
        
        # 添加标记
        if value_col:
            # 创建标记簇
            marker_cluster = MarkerCluster().add_to(m)
            
            # 归一化值用于颜色和大小
            min_val = df[value_col].min()
            max_val = df[value_col].max()
            range_val = max_val - min_val if max_val > min_val else 1
            
            # 添加带颜色的标记
            for idx, row in df.iterrows():
                val = row[value_col]
                normalized = (val - min_val) / range_val
                
                # 计算颜色 (红色表示高值，蓝色表示低值)
                color = f'#{int(255 * (1-normalized)):02x}{int(0):02x}{int(255 * normalized):02x}'
                
                # 标记大小从5到15
                radius = 5 + normalized * 10
                
                # 创建标记
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"{value_col}: {val:.2f}"
                ).add_to(marker_cluster)
            
            # 添加热力图
            heat_data = [[row[lat_col], row[lon_col], row[value_col]] for _, row in df.iterrows()]
            HeatMap(heat_data).add_to(m)
        else:
            # 简单标记
            for idx, row in df.iterrows():
                folium.Marker([row[lat_col], row[lon_col]]).add_to(m)
        
        # 保存地图
        if output_file:
            m.save(output_file)
            logger.info(f"位置地图已保存到 {output_file}")
        
        return m
    except Exception as e:
        logger.error(f"生成位置地图时出错: {str(e)}")
        raise

def create_interactive_dashboard(forecasts_df, locations_df, output_file=None):
    """
    创建交互式Plotly仪表板
    
    参数:
    - forecasts_df: 客流预测数据
    - locations_df: 位置评分数据
    - output_file: 输出HTML文件路径，如果为None则不保存
    
    返回:
    - fig: Plotly图形对象
    """
    try:
        logger.info("创建交互式仪表板...")
        
        # 创建2x2子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("客流预测趋势", "位置评分分布", "特征影响", "客流预测对比"),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. 客流预测趋势图 (假设forecasts_df有日期和预测列)
        if 'date' in forecasts_df.columns and 'prediction' in forecasts_df.columns:
            forecasts_df = forecasts_df.sort_values('date')
            fig.add_trace(
                go.Scatter(
                    x=forecasts_df['date'],
                    y=forecasts_df['prediction'],
                    mode='lines+markers',
                    name='预测客流'
                ),
                row=1, col=1
            )
            
            # 如果有实际值，也绘制实际客流
            if 'actual' in forecasts_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecasts_df['date'],
                        y=forecasts_df['actual'],
                        mode='lines+markers',
                        name='实际客流',
                        line=dict(dash='dot')
                    ),
                    row=1, col=1
                )
        
        # 2. 位置评分条形图
        if 'location_id' in locations_df.columns and 'predicted_score' in locations_df.columns:
            # 取前15个位置
            top_locations = locations_df.sort_values('predicted_score', ascending=False).head(15)
            
            fig.add_trace(
                go.Bar(
                    x=top_locations['location_id'],
                    y=top_locations['predicted_score'],
                    name='位置评分'
                ),
                row=1, col=2
            )
        
        # 3. 特征影响图 (假设locations_df有特征列)
        # 选择一些关键特征
        key_features = [col for col in locations_df.columns 
                        if col.startswith('poi_') or 'distance_' in col or 'density' in col]
        if key_features:
            # 计算每个特征的平均值
            feature_means = locations_df[key_features].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(
                    x=feature_means.index,
                    y=feature_means.values,
                    name='特征均值'
                ),
                row=2, col=1
            )
        
        # 4. 预测值对比散点图
        if 'actual' in forecasts_df.columns and 'prediction' in forecasts_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecasts_df['actual'],
                    y=forecasts_df['prediction'],
                    mode='markers',
                    name='预测 vs 实际',
                    marker=dict(size=8, opacity=0.6)
                ),
                row=2, col=2
            )
            
            # 添加对角线 (理想预测线)
            min_val = min(forecasts_df['actual'].min(), forecasts_df['prediction'].min())
            max_val = max(forecasts_df['actual'].max(), forecasts_df['prediction'].max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='理想预测',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title_text="酒店选址与客流预测仪表板",
            title_font_size=20,
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="客流量", row=1, col=1)
        
        fig.update_xaxes(title_text="位置ID", row=1, col=2)
        fig.update_yaxes(title_text="评分", row=1, col=2)
        
        fig.update_xaxes(title_text="特征", row=2, col=1)
        fig.update_yaxes(title_text="值", row=2, col=1)
        
        fig.update_xaxes(title_text="实际值", row=2, col=2)
        fig.update_yaxes(title_text="预测值", row=2, col=2)
        
        # 保存仪表板
        if output_file:
            fig.write_html(output_file)
            logger.info(f"交互式仪表板已保存到 {output_file}")
        
        return fig
    except Exception as e:
        logger.error(f"创建交互式仪表板时出错: {str(e)}")
        raise