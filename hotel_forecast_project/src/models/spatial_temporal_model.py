import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpatialAttention(nn.Module):
    """
    空间注意力机制 - 优化版本
    根据研究《Spatial Attention Mechanisms in Deep Learning》改进
    """
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 多层投影增强非线性表达能力
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
        # 初始化参数
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        # x: [batch, nodes, hidden_dim]
        energy = self.projection(x)  # [batch, nodes, 1]
        
        # 使用softmax生成权重
        weights = F.softmax(energy, dim=1)  # [batch, nodes, 1]
        
        # 添加额外的正则化项
        # 使用dropout防止过拟合
        weights = F.dropout(weights, p=0.1, training=self.training)
        
        outputs = weights * x  # [batch, nodes, hidden_dim]
        
        return outputs, weights

class TemporalAttention(nn.Module):
    """
    时间注意力机制 - 优化版本
    根据研究《Temporal Attention for Time Series Forecasting》改进
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 多层投影增强时间注意力能力
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
        # 初始化参数
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        # x: [batch, time_steps, hidden_dim]
        energy = self.projection(x)  # [batch, time_steps, 1]
        
        # 使用softmax生成权重，增加温度参数使分布更平滑
        temperature = 0.5
        weights = F.softmax(energy / temperature, dim=1)  # [batch, time_steps, 1]
        
        # 使用dropout防止过拟合
        weights = F.dropout(weights, p=0.1, training=self.training)
        
        outputs = weights * x  # [batch, time_steps, hidden_dim]
        return outputs, weights

class GraphConvLayer(nn.Module):
    """
    图卷积层 - 优化版本
    基于《Semi-Supervised Classification with Graph Convolutional Networks》
    增加了残差连接和归一化
    """
    def __init__(self, in_features, out_features, use_residual=True, dropout=0.1):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.use_residual = use_residual
        
        # 残差连接变换
        if use_residual:
            if in_features != out_features:
                self.residual_transform = nn.Linear(in_features, out_features)
            else:
                self.residual_transform = nn.Identity()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)
        
        # 参数重置
        self.reset_parameters()
        
    def reset_parameters(self):
        """参数初始化（Xavier均匀初始化）"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        # x: [batch, nodes, in_features]
        # adj: [batch, nodes, nodes] or [nodes, nodes]
        
        # 如果adj没有batch维度，添加一个
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).repeat(x.size(0), 1, 1)
            
        # 图卷积操作
        support = torch.matmul(x, self.weight)  # [batch, nodes, out_features]
        output = torch.matmul(adj, support)  # [batch, nodes, out_features]
        output = output + self.bias  # [batch, nodes, out_features]
        
        # 应用残差连接
        if self.use_residual:
            residual = self.residual_transform(x)
            output = output + residual
        
        # 应用层归一化
        output = self.layer_norm(output)
        
        # 应用激活函数和Dropout
        output = F.relu(output)
        output = self.dropout(output)
        
        return output

class PositionalEncoding(nn.Module):
    """
    位置编码 - 用于增强序列数据的位置信息
    基于Transformer架构《Attention Is All You Need》
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册缓冲区（不是模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x

class SpatialTemporalModel(nn.Module):
    """
    空间时间注意力机制模型 - 优化版本
    集成了多种最新技术提升预测性能
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, seq_len, dropout=0.1, n_gnn_layers=2):
        super(SpatialTemporalModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.dropout = dropout
        self.n_gnn_layers = n_gnn_layers
        
        # 空间嵌入 - 初始化优化
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim) / np.sqrt(hidden_dim))
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # 特征转换
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 多层GNN
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, use_residual=True, dropout=dropout)
            for _ in range(n_gnn_layers)
        ])
        
        # GRU层 - 双向增强时序处理能力
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim // 2,  # 双向，每个方向隐藏维度减半
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if n_gnn_layers > 1 else 0
        )
        
        # 注意力机制
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 输出层 - 多层增强表达能力
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        
    def create_default_adjacency(self, device):
        """创建默认邻接矩阵"""
        # 创建完全连接图
        adj = torch.ones(self.num_nodes, self.num_nodes, device=device)
        # 加入自环
        adj = adj + torch.eye(self.num_nodes, device=device)
        # 归一化
        adj = adj / adj.sum(dim=1, keepdim=True)
        return adj
        
    def forward(self, x, adj=None, node_idx=None):
        """
        前向传播
        
        参数:
        - x: [batch, seq_len, input_dim] - 输入特征序列
        - adj: [nodes, nodes] 或 [batch, nodes, nodes] - 空间邻接矩阵（可选）
        - node_idx: 要预测的节点索引列表或张量（可选）
        
        返回:
        - output: [batch, output_dim] - 模型输出
        """
        batch_size = x.size(0)
        device = x.device
        
        # 特征转换
        x = self.feature_proj(x)  # [batch, seq_len, hidden_dim]
        
        # 添加位置编码增强序列信息
        x = self.positional_encoding(x)
        
        # 时间编码
        x, _ = self.gru(x)  # [batch, seq_len, hidden_dim]
        
        # 时间注意力
        temporal_out, temporal_weights = self.temporal_attention(x)  # [batch, seq_len, hidden_dim]
        temporal_out = torch.sum(temporal_out, dim=1)  # [batch, hidden_dim]
        
        # 空间处理
        # 检查node_idx是否为有效值
        if isinstance(node_idx, torch.Tensor) and node_idx.numel() > 0:
            # 处理节点索引张量
            if node_idx.dim() == 1:
                node_emb = self.node_embeddings[node_idx]  # [batch, hidden_dim]
            else:
                # 如果每个样本有不同索引，单独处理每个样本
                node_emb = torch.stack([
                    self.node_embeddings[idx[0]]
                    for idx in node_idx
                ])  # [batch, hidden_dim]
        elif isinstance(node_idx, list) and node_idx:
            # 处理节点索引列表
            node_emb = torch.stack([
                self.node_embeddings[idx]
                for idx in node_idx
            ])  # [batch, hidden_dim]
        else:
            # 如果没有提供node_idx或它是空的
            # 创建默认邻接矩阵（如果未提供）
            if adj is None:
                adj = self.create_default_adjacency(device)
                
            # 展开空间嵌入为批次
            node_features = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 应用多层GNN
            for gnn_layer in self.gnn_layers:
                node_features = gnn_layer(node_features, adj)
            
            # 空间注意力
            attended_nodes, spatial_weights = self.spatial_attention(node_features)
            node_emb = torch.sum(attended_nodes, dim=1)  # [batch, hidden_dim]
        
        # 组合时空特征
        combined = torch.cat([temporal_out, node_emb], dim=1)  # [batch, hidden_dim*2]
        
        # 特征融合
        fused = self.fusion_layer(combined)  # [batch, hidden_dim]
        
        # 输出预测
        output = self.output_layer(fused)  # [batch, output_dim]
        
        return output
    
    def predict(self, x, adj=None, node_idx=None):
        """
        使用模型进行预测
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, adj, node_idx)

class SpatialTemporalTrainer:
    """
    空间时间注意力模型训练器
    """
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 weight_decay=1e-5,
                 device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器 - 使用AdamW
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器 - 余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=10,
            eta_min=learning_rate / 10
        )
        
        # 损失函数 - 组合损失
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.msle_weight = 0.2  # MSLE权重
        
        logger.info(f"模型初始化完成，使用设备: {self.device}")
    
    def msle_loss(self, y_pred, y_true, epsilon=1e-8):
        """
        计算MSLE损失，对大误差更敏感
        """
        return torch.mean(torch.pow(torch.log(y_pred + epsilon) - torch.log(y_true + epsilon), 2))
    
    def combined_loss(self, y_pred, y_true):
        """
        结合多种损失函数，增强模型对不同规模误差的敏感度
        """
        mse_loss = self.mse_loss(y_pred, y_true)
        smooth_l1_loss = self.smooth_l1_loss(y_pred, y_true)
        msle_loss = self.msle_loss(y_pred, y_true)
        
        # 组合损失
        combined = 0.5 * mse_loss + 0.3 * smooth_l1_loss + 0.2 * msle_loss
        return combined
    
    def train_epoch(self, dataloader, adj=None):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        batch_count = 0
        
        for batch_idx, (x, y, node_indices) in enumerate(dataloader):
            batch_count += 1
            batch_size = x.size(0)
            total_samples += batch_size
            
            x, y = x.to(self.device), y.to(self.device)
            if adj is not None:
                adj = adj.to(self.device)
                
            # 将节点索引移至当前设备
            if isinstance(node_indices, torch.Tensor) and node_indices.numel() > 0:
                node_indices = node_indices.to(self.device)
                
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(x, adj, node_indices)
            
            # 计算损失
            loss = self.combined_loss(outputs, y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * batch_size
            
            if batch_idx % 10 == 0:
                logger.info(f"批次 {batch_idx}/{len(dataloader)}, 损失: {loss.item():.6f}")
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return avg_loss
    
    def validate(self, dataloader, adj=None):
        """
        在验证集上评估模型
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y, node_indices in dataloader:
                batch_size = x.size(0)
                total_samples += batch_size
                
                x, y = x.to(self.device), y.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                    
                # 将节点索引移至当前设备
                if isinstance(node_indices, torch.Tensor) and node_indices.numel() > 0:
                    node_indices = node_indices.to(self.device)
                    
                # 前向传播
                outputs = self.model(x, adj, node_indices)
                
                # 计算损失
                loss = self.combined_loss(outputs, y)
                total_loss += loss.item() * batch_size
                
                # 收集预测和真实值
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # 计算评估指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        return metrics
    
    def train(self, train_dataloader, valid_dataloader, epochs, adj=None, patience=10, model_path=None):
        """
        完整训练流程
        """
        logger.info(f"开始训练，总epochs: {epochs}")
        
        best_val_loss = float('inf')
        no_improvement = 0
        
        train_losses = []
        valid_losses = []
        valid_metrics = []
        best_epoch = -1
        best_state_dict = None
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_dataloader, adj)
            train_losses.append(train_loss)
            
            # 验证
            metrics = self.validate(valid_dataloader, adj)
            valid_loss = metrics['loss']
            valid_losses.append(valid_loss)
            valid_metrics.append(metrics)
            
            # 更新学习率调度器
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            logger.info(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f} - 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}, "
                      f"验证MAE: {metrics['mae']:.4f}, 验证RMSE: {metrics['rmse']:.4f}, 验证R²: {metrics['r2']:.4f}")
            
            # 早停
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_epoch = epoch
                best_state_dict = self.model.state_dict().copy()
                no_improvement = 0
                
                # 保存最佳模型
                if model_path:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'valid_loss': valid_loss,
                        'valid_metrics': metrics
                    }
                    torch.save(checkpoint, model_path)
                    logger.info(f"模型已保存到 {model_path}")
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                logger.info(f"早停: {patience} 个epoch没有改善")
                break
        
        # 加载最佳模型
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(f"已加载最佳模型 (Epoch {best_epoch+1})")
        
        # 返回训练历史
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'valid_metrics': valid_metrics,
            'best_epoch': best_epoch
        }
    
    def predict(self, x, adj=None, node_idx=None):
        """
        使用模型进行预测
        """
        self.model.eval()
        
        # 将输入转换为张量
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        # 处理邻接矩阵
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.FloatTensor(adj)
            adj = adj.to(self.device)
            
        # 处理节点索引
        if isinstance(node_idx, list) and node_idx:
            node_idx = torch.LongTensor(node_idx).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, adj, node_idx)
            
        return outputs.cpu().numpy()

class SpatialTemporalDataset(torch.utils.data.Dataset):
    """
    空间时间数据集 - 优化版本
    处理各种形式的输入数据
    """
    def __init__(self, features, targets, node_indices=None, transform=None):
        """
        初始化数据集
        
        参数:
        - features: [samples, seq_len, features] 形状的特征序列
        - targets: [samples, output_dim] 形状的目标值
        - node_indices: 节点索引列表，可以是None、列表或张量
        - transform: 可选的数据变换函数
        """
        # 转换为numpy数组以确保兼容性
        if isinstance(features, torch.Tensor):
            self.features = features.numpy()
        else:
            self.features = np.asarray(features, dtype=np.float32)
            
        if isinstance(targets, torch.Tensor):
            self.targets = targets.numpy()
        else:
            self.targets = np.asarray(targets, dtype=np.float32)
            
        # 处理节点索引
        if node_indices is None:
            # 如果没有节点索引，使用空张量
            self.node_indices = [torch.tensor([], dtype=torch.long)] * len(features)
            self.use_node_indices = False
        else:
            # 转换节点索引为适当格式
            if isinstance(node_indices, (list, tuple)):
                self.node_indices = [
                    torch.tensor([idx], dtype=torch.long) if idx is not None else torch.tensor([], dtype=torch.long)
                    for idx in node_indices
                ]
            elif isinstance(node_indices, np.ndarray):
                if node_indices.ndim == 1:
                    self.node_indices = [
                        torch.tensor([idx], dtype=torch.long) if idx >= 0 else torch.tensor([], dtype=torch.long)
                        for idx in node_indices
                    ]
                else:
                    self.node_indices = [
                        torch.tensor(row, dtype=torch.long) if np.all(row >= 0) else torch.tensor([], dtype=torch.long)
                        for row in node_indices
                    ]
            elif isinstance(node_indices, torch.Tensor):
                if node_indices.dim() == 1:
                    self.node_indices = [
                        torch.tensor([idx], dtype=torch.long) if idx >= 0 else torch.tensor([], dtype=torch.long)
                        for idx in node_indices
                    ]
                else:
                    self.node_indices = [
                        torch.tensor(row, dtype=torch.long) if torch.all(row >= 0) else torch.tensor([], dtype=torch.long)
                        for row in node_indices
                    ]
            else:
                raise ValueError(f"不支持的节点索引类型: {type(node_indices)}")
                
            self.use_node_indices = True
            
        # 数据变换
        self.transform = transform
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        x = self.features[idx]
        y = self.targets[idx]
        node_idx = self.node_indices[idx]
        
        # 应用数据变换
        if self.transform:
            x = self.transform(x)
            
        return torch.FloatTensor(x), torch.FloatTensor(y), node_idx

def prepare_sequence_data(df, feature_cols, target_cols, seq_len=7, stride=1, node_col=None, scaler=None):
    """
    准备序列数据
    
    参数:
    - df: 输入DataFrame
    - feature_cols: 特征列名列表
    - target_cols: 目标列名列表
    - seq_len: 序列长度
    - stride: 滑动窗口步长
    - node_col: 节点标识列（可选）
    - scaler: 特征标准化器（可选），如果为None则创建新的
    
    返回:
    - X: 特征序列，形状为 [samples, seq_len, features]
    - y: 目标值，形状为 [samples, targets]
    - node_indices: 节点索引列表（如果提供了node_col）
    - scaler: 使用的StandardScaler对象
    """
    # 确保所有特征列都存在
    for col in feature_cols + target_cols:
        if col not in df.columns:
            raise ValueError(f"列 {col} 不在DataFrame中")
    
    # 将特征和目标转换为numpy数组
    X_data = df[feature_cols].values
    y_data = df[target_cols].values
    
    # 特征标准化
    if scaler is None:
        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_data)
    else:
        X_data = scaler.transform(X_data)
    
    # 重新整形回原来的形状
    X_data = X_data.reshape(df.shape[0], -1)
    
    # 如果提供了节点列，获取唯一节点并创建映射
    node_indices = None
    if node_col and node_col in df.columns:
        nodes = sorted(df[node_col].unique())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        node_indices = df[node_col].map(node_to_idx).values
    
    # 创建序列数据
    X, y, seq_node_indices = [], [], []
    
    for i in range(0, len(df) - seq_len + 1, stride):
        X.append(X_data[i:i+seq_len].reshape(seq_len, -1))
        y.append(y_data[i+seq_len-1])  # 使用序列最后一个时间点的目标
        
        if node_indices is not None:
            seq_node_indices.append(node_indices[i+seq_len-1])
    
    return np.array(X), np.array(y), seq_node_indices if node_indices is not None else None, scaler

def create_adjacency_matrix(df, node_col, distance_col=None, threshold=None, weighted=True):
    """
    创建邻接矩阵
    
    参数:
    - df: 包含节点关系的DataFrame
    - node_col: 节点标识列名
    - distance_col: 距离列名（可选，用于加权图）
    - threshold: 连接阈值（可选，如果提供则小于阈值的节点才连接）
    - weighted: 是否使用加权图
    
    返回:
    - adj_matrix: 邻接矩阵，形状为 [nodes, nodes]
    - node_to_idx: 节点到索引的映射
    """
    # 获取唯一节点并创建映射
    nodes = sorted(df[node_col].unique())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # 初始化邻接矩阵
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    if distance_col is not None and distance_col in df.columns:
        # 基于距离列创建邻接矩阵
        for _, row in df.iterrows():
            i = node_to_idx[row[node_col]]
            
            for _, other_row in df.iterrows():
                if row[node_col] == other_row[node_col]:
                    continue
                    
                j = node_to_idx[other_row[node_col]]
                
                # 计算距离
                distance = abs(row[distance_col] - other_row[distance_col])
                
                # 如果有阈值，检查是否小于阈值
                if threshold is not None and distance > threshold:
                    continue
                
                if weighted:
                    # 使用距离的倒数作为权重，避免除以0
                    adj_matrix[i, j] = 1.0 / (distance + 1e-6)
                else:
                    adj_matrix[i, j] = 1.0
    else:
        # 如果没有距离列，创建完全连接图
        adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    
    # 归一化邻接矩阵
    if np.sum(adj_matrix) > 0:  # 避免除零
        # 对称归一化
        rowsum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_matrix = np.matmul(np.matmul(d_mat_inv_sqrt, adj_matrix), d_mat_inv_sqrt)
    
    return adj_matrix, node_to_idx

def load_model(model_path, input_dim, hidden_dim, output_dim, num_nodes, seq_len, device=None):
    """
    加载保存的模型
    
    参数:
    - model_path: 模型文件路径
    - input_dim: 输入特征维度
    - hidden_dim: 隐藏层维度
    - output_dim: 输出维度
    - num_nodes: 空间节点数
    - seq_len: 序列长度
    - device: 设备（'cuda'或'cpu'）
    
    返回:
    - model: 加载的模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型实例
    model = SpatialTemporalModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_nodes=num_nodes,
        seq_len=seq_len
    ).to(device)
    
    # 加载状态字典
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果是完整检查点，提取模型状态字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 直接加载模型状态字典
        model.load_state_dict(checkpoint)
    
    return model