import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
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
    空间注意力机制
    """
    def __init__(self, hidden_dim):
        super(SpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [batch, nodes, hidden_dim]
        energy = self.projection(x)  # [batch, nodes, 1]
        weights = F.softmax(energy, dim=1)  # [batch, nodes, 1]
        outputs = weights * x  # [batch, nodes, hidden_dim]
        return outputs, weights

class TemporalAttention(nn.Module):
    """
    时间注意力机制
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: [batch, time_steps, hidden_dim]
        energy = self.projection(x)  # [batch, time_steps, 1]
        weights = F.softmax(energy, dim=1)  # [batch, time_steps, 1]
        outputs = weights * x  # [batch, time_steps, hidden_dim]
        return outputs, weights

class GraphConvLayer(nn.Module):
    """
    图卷积层
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        # x: [batch, nodes, in_features]
        # adj: [batch, nodes, nodes] or [nodes, nodes]
        
        # 如果adj没有batch维度，添加一个
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).repeat(x.size(0), 1, 1)
            
        support = torch.matmul(x, self.weight)  # [batch, nodes, out_features]
        output = torch.matmul(adj, support)  # [batch, nodes, out_features]
        output = output + self.bias  # [batch, nodes, out_features]
        
        return output

class SpatialTemporalModel(nn.Module):
    """
    空间时间注意力机制模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, seq_len, dropout=0.1):
        super(SpatialTemporalModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.dropout = dropout
        
        # 空间嵌入
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim), requires_grad=True)
        
        # 特征转换
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN层
        self.gcn = GraphConvLayer(hidden_dim, hidden_dim)
        
        # GRU层
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 注意力机制
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, adj=None, node_idx=None):
        # x: [batch, seq_len, input_dim]
        # adj: [nodes, nodes] - 空间邻接矩阵（可选）
        # node_idx: 要预测的节点索引列表（可选），可以是空张量
        
        batch_size = x.size(0)
        
        # 特征转换
        x = self.feature_proj(x)  # [batch, seq_len, hidden_dim]
        
        # 时间编码
        x, _ = self.gru(x)  # [batch, seq_len, hidden_dim]
        
        # 时间注意力
        temporal_out, temporal_weights = self.temporal_attention(x)  # [batch, seq_len, hidden_dim]
        temporal_out = torch.sum(temporal_out, dim=1)  # [batch, hidden_dim]
        
        # 空间处理
        # 检查node_idx是否为有效值
        if node_idx is not None and (isinstance(node_idx, list) or isinstance(node_idx, np.ndarray) or 
                                    (isinstance(node_idx, torch.Tensor) and node_idx.numel() > 0)):
            # 获取特定节点的嵌入
            if isinstance(node_idx, list) or isinstance(node_idx, np.ndarray):
                node_emb = self.node_embeddings[node_idx]  # [batch, hidden_dim]
            elif isinstance(node_idx, torch.Tensor):
                if node_idx.dim() == 1:
                    node_emb = self.node_embeddings[node_idx]  # [batch, hidden_dim]
                elif node_idx.dim() == 2:
                    # 使用每个样本的特定索引
                    node_emb = torch.stack([self.node_embeddings[idx[0]] for idx in node_idx])  # [batch, hidden_dim]
                else:
                    # 回退到默认行为
                    node_emb = self.node_embeddings[0].unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_dim]
            else:
                node_emb = self.node_embeddings[node_idx].unsqueeze(0).expand(batch_size, -1)  # [batch, hidden_dim]
        else:
            # 构建默认邻接矩阵（如果未提供）
            if adj is None:
                # 创建完全连接图
                adj = torch.ones(self.num_nodes, self.num_nodes, device=x.device)
                # 归一化
                adj = adj / adj.sum(dim=1, keepdim=True)
                
            # GCN传播
            node_features = self.gcn(self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1), adj)  # [batch, num_nodes, hidden_dim]
            
            # 空间注意力
            node_emb, spatial_weights = self.spatial_attention(node_features)  # [batch, num_nodes, hidden_dim]
            node_emb = torch.sum(node_emb, dim=1)  # [batch, hidden_dim]
        
        # 组合时空特征
        combined = torch.cat([temporal_out, node_emb], dim=1)  # [batch, hidden_dim*2]
        
        # 输出预测
        output = self.output_layer(combined)  # [batch, output_dim]
        
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
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 损失函数 - MSE + MSLE 组合
        self.criterion = nn.MSELoss()
        self.msle_weight = 0.2  # MSLE权重
        
        logger.info(f"模型初始化完成，使用设备: {self.device}")
    
    def msle_loss(self, y_pred, y_true, epsilon=1e-8):
        """
        计算MSLE损失，对大误差更敏感
        """
        return torch.mean(torch.pow(torch.log(y_pred + epsilon) - torch.log(y_true + epsilon), 2))
    
    def combined_loss(self, y_pred, y_true):
        """
        结合MSE和MSLE的损失函数
        """
        mse = self.criterion(y_pred, y_true)
        msle = self.msle_loss(y_pred, y_true)
        return (1 - self.msle_weight) * mse + self.msle_weight * msle
    
    def train_epoch(self, dataloader, adj=None):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        for batch_idx, (x, y, node_indices) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if adj is not None:
                adj = adj.to(self.device)
                
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(x, adj, node_indices)
            
            # 计算损失
            loss = self.combined_loss(outputs, y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"批次 {batch_idx}/{len(dataloader)}, 损失: {loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader, adj=None):
        """
        在验证集上评估模型
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y, node_indices in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                if adj is not None:
                    adj = adj.to(self.device)
                    
                # 前向传播
                outputs = self.model(x, adj, node_indices)
                
                # 计算损失
                loss = self.combined_loss(outputs, y)
                total_loss += loss.item()
                
                # 收集预测和真实值
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # 计算评估指标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        
        avg_loss = total_loss / len(dataloader)
        
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
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_dataloader, adj)
            train_losses.append(train_loss)
            
            # 验证
            metrics = self.validate(valid_dataloader, adj)
            valid_loss = metrics['loss']
            valid_losses.append(valid_loss)
            valid_metrics.append(metrics)
            
            # 学习率调整
            self.scheduler.step(valid_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}, "
                      f"验证MAE: {metrics['mae']:.4f}, 验证RMSE: {metrics['rmse']:.4f}, 验证R²: {metrics['r2']:.4f}")
            
            # 早停
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                no_improvement = 0
                
                # 保存最佳模型
                if model_path:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'valid_loss': valid_loss,
                        'valid_metrics': metrics
                    }, model_path)
                    logger.info(f"模型已保存到 {model_path}")
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                logger.info(f"早停: {patience} 个epoch没有改善")
                break
        
        # 加载最佳模型（如果保存了）
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"已加载最佳模型 (Epoch {checkpoint['epoch']+1})")
        
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'valid_metrics': valid_metrics
        }
    
    def predict(self, x, adj=None, node_idx=None):
        """
        使用模型进行预测
        """
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)
            
        with torch.no_grad():
            outputs = self.model(x, adj, node_idx)
            
        return outputs.cpu().numpy()

class SpatialTemporalDataset(torch.utils.data.Dataset):
    """
    空间时间数据集
    """
    def __init__(self, features, targets, node_indices=None, seq_len=7, transform=None):
        self.features = features  # [samples, seq_len, features]
        self.targets = targets    # [samples, output_dim]
        
        # 如果没有提供节点索引，使用空张量代替None
        if node_indices is None:
            self.use_node_indices = False
            self.node_indices = [0] * len(features)  # 使用0代替None
        else:
            self.use_node_indices = True
            self.node_indices = node_indices
            
        self.seq_len = seq_len
        self.transform = transform  # 可选的数据转换
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        
        # 如果不使用节点索引，返回一个空张量
        if not self.use_node_indices:
            node_idx = torch.tensor([])
        else:
            # 确保节点索引是张量
            node_idx = torch.tensor([self.node_indices[idx]])
        
        if self.transform:
            x = self.transform(x)
            
        return torch.FloatTensor(x), torch.FloatTensor(y), node_idx

def prepare_sequence_data(df, feature_cols, target_cols, seq_len=7, stride=1, node_col=None):
    """
    准备序列数据
    
    参数:
    - df: 输入DataFrame
    - feature_cols: 特征列名列表
    - target_cols: 目标列名列表
    - seq_len: 序列长度
    - stride: 滑动窗口步长
    - node_col: 节点标识列（可选）
    
    返回:
    - X: 特征序列，形状为 [samples, seq_len, features]
    - y: 目标值，形状为 [samples, targets]
    - node_indices: 节点索引列表（如果提供了node_col）
    """
    # 将特征和目标转换为numpy数组
    X_data = df[feature_cols].values
    y_data = df[target_cols].values
    
    # 如果提供了节点列，获取唯一节点并创建映射
    node_indices = None
    if node_col and node_col in df.columns:
        nodes = df[node_col].unique()
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        node_indices = df[node_col].map(node_to_idx).values
    
    # 创建序列数据
    X, y, seq_node_indices = [], [], []
    
    for i in range(0, len(df) - seq_len + 1, stride):
        X.append(X_data[i:i+seq_len])
        y.append(y_data[i+seq_len-1])  # 使用序列最后一个时间点的目标
        
        if node_indices is not None:
            seq_node_indices.append(node_indices[i+seq_len-1])
    
    return np.array(X), np.array(y), seq_node_indices if node_indices is not None else None

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
    nodes = df[node_col].unique()
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
                    # 使用距离的倒数作为权重
                    adj_matrix[i, j] = 1.0 / (distance + 1e-6)
                else:
                    adj_matrix[i, j] = 1.0
    else:
        # 如果没有距离列，创建完全连接图
        adj_matrix = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
    
    # 归一化
    if adj_matrix.sum() > 0:  # 避免除零
        adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    
    return adj_matrix, node_to_idx 