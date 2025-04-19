# 酒店选址预测项目笔记本

本目录包含用于酒店选址预测的 Jupyter 笔记本文件。

## 修复说明

原始笔记本 `01_basic_model.ipynb` 在处理日期数据时遇到错误，错误消息为：
```
day is out of range for month, at position 132
```

这是因为在源数据中有无效的日期（例如：2月31日）。

此外，模型训练部分也遇到了错误：
```
TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>
```

这是因为 `SpatialTemporalDataset` 类在处理没有节点索引的数据时使用了 `None` 值。

## 修复文件

我们提供了以下修复文件：

1. `fix_preprocessing.py` - 修复数据预处理问题的独立脚本
    - 脚本重新加载预处理模块并添加错误处理
    - 将预处理后的数据保存到 `data/processed/` 目录

2. 已修改 `src/models/spatial_temporal_model.py`:
    - 更新了 `SpatialTemporalDataset` 类处理 `None` 节点索引的方式
    - 修改了 `forward` 方法以正确处理空张量节点索引
    - 解决了数据加载器中的批处理错误

3. `test_model.py` - 测试修复后的模型
    - 验证模型可以正确处理没有节点索引的数据

## 使用方法

1. 首先运行修复脚本生成预处理数据：
   ```
   python notebooks/fix_preprocessing.py
   ```

2. 然后打开修复版本的笔记本：
   ```
   jupyter notebook notebooks/01_basic_model.ipynb
   ```

3. 使用已预处理的数据文件，避免重新运行处理步骤

## 主要修复内容

1. **日期处理修复**:
   - 修复了日期处理逻辑，使用 `errors='coerce'` 参数和适当的回退机制处理无效日期
   - 优化了数据流程，将预处理和特征工程步骤分离

2. **模型修复**:
   - 用空张量替换 `None` 值，解决了批处理错误
   - 更新了模型的 `forward` 方法，增加了对张量输入的正确处理
   - 增加了模型健壮性，适应不同类型的节点索引输入

3. **其他改进**:
   - 添加了详细的日志记录和错误处理
   - 提供更多用于调试的信息输出
   - 增加了测试脚本验证修复有效性 