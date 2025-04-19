import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the fixed modules
from src.models.spatial_temporal_model import SpatialTemporalModel, SpatialTemporalTrainer, SpatialTemporalDataset

print("创建测试数据...")
# Create some sample data
batch_size = 4
seq_len = 7
input_dim = 10
output_dim = 1
num_samples = 20

# Create random features and targets
features = np.random.randn(num_samples, seq_len, input_dim)
targets = np.random.randn(num_samples, output_dim)

print("测试不包含节点索引的数据加载器...")
# Create dataset without node indices
dataset = SpatialTemporalDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model
model = SpatialTemporalModel(
    input_dim=input_dim,
    hidden_dim=32,
    output_dim=output_dim,
    num_nodes=100,
    seq_len=seq_len
)

# Try loading a batch
for batch_idx, (x, y, node_indices) in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  特征形状: {x.shape}")
    print(f"  目标形状: {y.shape}")
    print(f"  节点索引形状: {node_indices.shape}")
    print(f"  节点索引类型: {node_indices.dtype}")
    
    # Try a forward pass
    output = model(x, adj=None, node_idx=node_indices)
    print(f"  输出形状: {output.shape}")
    
    # Only process one batch for this test
    break

print("测试成功完成!") 