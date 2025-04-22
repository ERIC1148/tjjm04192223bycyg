"""
图注意力网络(Graph Attention Networks)模型实现

本模块提供基于GAT的空间注意力模型，用于增强酒店选址与客流预测系统的空间建模能力。
参考论文: Graph Attention Networks (Veličković et al., ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GraphAttentionLayer(nn.Module):
    """
    图注意力层，基于GAT论文(Veličković et al., 2018)
    
    特点：
    - 使用注意力机制聚合邻居节点的信息
    - 允许为不同邻居分配不同的权重
    - 支持多头注意力
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        """
        初始化图注意力层
        
        参数:
        - in_features: 输入特征维度
        - out_features: 输出特征维度
        - dropout: dropout概率
        - alpha: LeakyReLU负斜率
        - concat: 是否对多头注意力结果进行拼接(True)或求平均(False)
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 可学习的参数
        # 特征变换矩阵W
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        # 注意力机制参数a
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        
        # 初始化参数
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
        # LeakyReLU用于注意力计算
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
        - x: 形状为[batch, nodes, in_features]的节点特征
        - adj: 形状为[batch, nodes, nodes]的邻接矩阵
        
        返回:
        - 形状为[batch, nodes, out_features]的新节点特征
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # 线性变换所有节点特征
        h = torch.matmul(x, self.W)  # [batch, nodes, out_features]
        
        # 准备注意力计算
        # 对每对节点(i,j)，生成拼接特征[h_i || h_j]
        a_input = torch.cat([
            h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, self.out_features),
            h.repeat(1, num_nodes, 1)
        ], dim=2).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)
        # a_input: [batch, nodes, nodes, 2*out_features]
        
        # 计算注意力系数 e_ij = a^T(Wh_i || Wh_j)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # e: [batch, nodes, nodes]
        
        # 屏蔽非邻居节点的注意力
        zero_vec = -9e15 * torch.ones_like(e)
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 归一化注意力系数（对每个节点的所有邻居进行softmax）
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 使用注意力系数加权节点特征
        h_prime = torch.matmul(attention, h)  # [batch, nodes, out_features]
        
        if self.concat:
            # 如果是用于拼接的中间层，应用非线性激活
            return F.elu(h_prime)
        else:
            # 如果是输出层，不应用非线性激活
            return h_prime

class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力层
    
    通过多个独立的注意力机制捕捉不同方面的信息，然后合并结果
    """
    def __init__(self, in_features, out_features, n_heads=8, dropout=0.1, alpha=0.2):
        """
        初始化多头图注意力层
        
        参数:
        - in_features: 输入特征维度
        - out_features: 输出特征维度
        - n_heads: 注意力头数量
        - dropout: dropout概率
        - alpha: LeakyReLU负斜率
        """
        super(MultiHeadGraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # 多个注意力头
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=out_features,
                dropout=dropout,
                alpha=alpha,
                concat=True  # 中间层结果用于拼接
            ) for _ in range(n_heads)
        ])
        
        # 输出层
        self.out_att = GraphAttentionLayer(
            in_features=out_features * n_heads,  # 输入是所有头的拼接结果
            out_features=out_features,
            dropout=dropout,
            alpha=alpha,
            concat=False  # 输出层结果不拼接
        )
        
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
        - x: 形状为[batch, nodes, in_features]的节点特征
        - adj: 形状为[batch, nodes, nodes]的邻接矩阵
        
        返回:
        - 形状为[batch, nodes, out_features]的新节点特征
        """
        # 独立处理每个注意力头
        head_outputs = [att(x, adj) for att in self.attentions]
        
        # 拼接多头的输出
        multi_head = torch.cat(head_outputs, dim=2)  # [batch, nodes, out_features*n_heads]
        
        # 应用dropout
        multi_head = F.dropout(multi_head, self.dropout, training=self.training)
        
        # 通过输出层获得最终输出
        output = self.out_att(multi_head, adj)
        
        return output

class SpatialAttentionGAT(nn.Module):
    """
    基于GAT的空间注意力模块
    
    使用图注意力网络捕捉空间实体之间的复杂关系，适用于酒店选址模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=8, n_layers=2, dropout=0.1):
        """
        初始化GAT空间注意力模块
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - output_dim: 输出特征维度
        - n_heads: 注意力头数量
        - n_layers: 图注意力层数量
        - dropout: dropout概率
        """
        super(SpatialAttentionGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多个GAT层
        self.gat_layers = nn.ModuleList()
        
        # 第一层，将输入特征映射到隐藏维度
        self.gat_layers.append(
            MultiHeadGraphAttention(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
        )
        
        # 中间层
        for i in range(1, n_layers-1):
            self.gat_layers.append(
                MultiHeadGraphAttention(hidden_dim, hidden_dim, n_heads=n_heads, dropout=dropout)
            )
        
        # 输出层
        self.gat_layers.append(
            MultiHeadGraphAttention(hidden_dim, output_dim, n_heads=n_heads, dropout=dropout)
        )
        
        # 层归一化，增强模型稳定性
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers-1)
        ] + [nn.LayerNorm(output_dim)])
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def create_adjacency_matrix(self, batch_size, num_nodes, device):
        """
        创建默认邻接矩阵（全连接图）
        
        参数:
        - batch_size: 批次大小
        - num_nodes: 节点数量
        - device: 设备(CPU/GPU)
        
        返回:
        - 形状为[batch_size, num_nodes, num_nodes]的默认邻接矩阵
        """
        # 创建完全连接图（排除自环）
        adj = torch.ones(num_nodes, num_nodes, device=device) - torch.eye(num_nodes, device=device)
        
        # 归一化邻接矩阵
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        # 扩展为批次形状
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adj
                    
    def forward(self, x, adj=None):
        """
        前向传播
        
        参数:
        - x: 形状为[batch, nodes, input_dim]的节点特征
        - adj: 形状为[batch, nodes, nodes]的邻接矩阵（可选）
        
        返回:
        - 形状为[batch, nodes, output_dim]的节点特征
        """
        batch_size, num_nodes, _ = x.size()
        device = x.device
        
        # 如果未提供邻接矩阵，创建默认的
        if adj is None:
            adj = self.create_adjacency_matrix(batch_size, num_nodes, device)
        elif adj.dim() == 2:
            # 如果提供的邻接矩阵没有批次维度，添加批次维度
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 应用多层GAT，每层之后使用残差连接和层归一化
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            residual = x
            x = gat_layer(x, adj)
            
            # 调整残差连接维度（如果需要）
            if i == len(self.gat_layers) - 1 and self.hidden_dim != self.output_dim:
                residual = nn.Linear(self.hidden_dim, self.output_dim, device=device)(residual)
                
            # 残差连接
            x = x + residual
            
            # 层归一化
            x = layer_norm(x)
            
            # 中间层应用非线性和dropout
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def compute_attention_weights(self, x, adj=None):
        """
        计算注意力权重，用于可视化和解释模型决策
        
        参数:
        - x: 形状为[batch, nodes, input_dim]的节点特征
        - adj: 形状为[batch, nodes, nodes]的邻接矩阵（可选）
        
        返回:
        - 列表，每个元素是一层的注意力权重，形状为[batch, n_heads, nodes, nodes]
        """
        batch_size, num_nodes, _ = x.size()
        device = x.device
        
        if adj is None:
            adj = self.create_adjacency_matrix(batch_size, num_nodes, device)
        elif adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        
        attention_weights = []
        current_x = x
        
        # 收集每层的注意力权重
        for gat_layer in self.gat_layers:
            # 这里简化了实现，实际需要修改GAT层以返回注意力权重
            # 此处仅为示例，真实实现需要按需修改
            layer_weights = []
            for head in gat_layer.attentions:
                # 计算注意力权重
                h = torch.matmul(current_x, head.W)  # [batch, nodes, out_features]
                
                a_input = torch.cat([
                    h.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, head.out_features),
                    h.repeat(1, num_nodes, 1)
                ], dim=2).view(batch_size, num_nodes, num_nodes, 2 * head.out_features)
                
                e = head.leakyrelu(torch.matmul(a_input, head.a).squeeze(3))
                
                zero_vec = -9e15 * torch.ones_like(e)
                attention = torch.where(adj > 0, e, zero_vec)
                attention = F.softmax(attention, dim=2)
                
                layer_weights.append(attention)
                
            # 合并多头注意力的权重
            layer_attention = torch.stack(layer_weights, dim=1)  # [batch, n_heads, nodes, nodes]
            attention_weights.append(layer_attention)
            
            # 更新节点特征，用于下一层
            next_x = []
            for head_idx, head in enumerate(gat_layer.attentions):
                head_attention = layer_weights[head_idx]
                head_out = torch.matmul(head_attention, h)  # [batch, nodes, out_features]
                next_x.append(F.elu(head_out))
            
            current_x = torch.cat(next_x, dim=2)
            
        return attention_weights


class GATSpatioTemporalModel(nn.Module):
    """
    基于GAT的时空模型
    
    结合GAT的空间建模能力和LSTM/GRU的时序建模能力，为酒店选址和预测提供更强大的模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, seq_len, 
                 n_heads=8, n_gat_layers=2, dropout=0.1):
        """
        初始化GAT时空模型
        
        参数:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层维度
        - output_dim: 输出特征维度
        - num_nodes: 空间节点数量
        - seq_len: 序列长度
        - n_heads: GAT注意力头数量
        - n_gat_layers: GAT层数量
        - dropout: dropout概率
        """
        super(GATSpatioTemporalModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        # 特征转换层
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 时间编码 - 位置编码
        self.position_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)
        
        # 时序模块 - 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 双向LSTM，每个方向隐藏维度减半
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 空间模块 - GAT
        self.spatial_gat = SpatialAttentionGAT(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_gat_layers,
            dropout=dropout
        )
        
        # 节点嵌入 - 可学习的空间表示
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim) / np.sqrt(hidden_dim))
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 输出层 - 融合时空特征
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x, adj=None, node_idx=None):
        """
        前向传播
        
        参数:
        - x: 形状为[batch, seq_len, input_dim]的序列特征
        - adj: 形状为[batch, nodes, nodes]或[nodes, nodes]的邻接矩阵（可选）
        - node_idx: 目标节点索引（可选）
        
        返回:
        - 形状为[batch, output_dim]的预测结果
        """
        batch_size = x.size(0)
        device = x.device
        
        # 特征转换
        x = self.feature_proj(x)  # [batch, seq_len, hidden_dim]
        
        # 应用位置编码
        x = self.position_encoding(x)
        
        # 时序处理 - LSTM
        x, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # 时间注意力
        temporal_out, temporal_weights = self.temporal_attention(x)  # [batch, seq_len, hidden_dim]
        temporal_out = torch.sum(temporal_out, dim=1)  # [batch, hidden_dim]
        
        # 空间处理
        # 检查是否提供了节点索引
        if isinstance(node_idx, torch.Tensor) and node_idx.numel() > 0:
            # 提取目标节点的嵌入
            if node_idx.dim() == 1:
                node_emb = self.node_embeddings[node_idx]  # [batch, hidden_dim]
            else:
                # 如果每个样本有不同索引，单独处理每个样本
                node_emb = torch.stack([
                    self.node_embeddings[idx[0]]
                    for idx in node_idx
                ])  # [batch, hidden_dim]
        elif isinstance(node_idx, list) and len(node_idx) > 0:
            # 处理节点索引列表
            node_emb = torch.stack([
                self.node_embeddings[idx]
                for idx in node_idx
            ])  # [batch, hidden_dim]
        else:
            # 如果没有提供目标节点索引，处理所有节点
            # 扩展节点嵌入为批次形式
            node_features = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, nodes, hidden_dim]
            
            # 创建或处理邻接矩阵
            if adj is None:
                # 使用默认邻接矩阵
                adj = torch.ones(self.num_nodes, self.num_nodes, device=device)
                adj = adj - torch.eye(self.num_nodes, device=device)  # 移除自环
                adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)  # 归一化
                
            # 应用GAT
            node_features = self.spatial_gat(node_features, adj)  # [batch, nodes, hidden_dim]
            
            # 全局池化得到图级表示
            node_emb = torch.mean(node_features, dim=1)  # [batch, hidden_dim]
        
        # 融合时空特征
        combined = torch.cat([temporal_out, node_emb], dim=1)  # [batch, hidden_dim*2]
        
        # 输出层
        output = self.output_layer(combined)  # [batch, output_dim]
        
        return output


class TemporalAttention(nn.Module):
    """
    时间注意力机制，用于捕捉序列中的重要时间步
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 多层注意力投影
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 形状为[batch, seq_len, hidden_dim]的序列特征
        
        返回:
        - outputs: 形状为[batch, seq_len, hidden_dim]的加权序列特征
        - weights: 形状为[batch, seq_len, 1]的注意力权重
        """
        # 计算注意力权重
        energy = self.projection(x)  # [batch, seq_len, 1]
        weights = F.softmax(energy, dim=1)  # [batch, seq_len, 1]
        
        # 加权序列特征
        outputs = weights * x  # [batch, seq_len, hidden_dim]
        
        return outputs, weights


class PositionalEncoding(nn.Module):
    """
    位置编码，为序列中的每个位置添加位置信息
    基于《Attention Is All You Need》论文中的实现
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为非学习参数的缓冲区
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        添加位置编码到输入张量
        
        参数:
        - x: 形状为[batch, seq_len, d_model]的输入张量
        
        返回:
        - 添加位置信息后的张量
        """
        # 截取需要长度的位置编码并添加到输入
        x = x + self.pe[:, :x.size(1), :]
        return x