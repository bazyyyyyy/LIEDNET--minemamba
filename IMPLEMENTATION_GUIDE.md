# VSSLocalBlock 模块替换实现指南

## 已完成的实现

### 1. ✅ 创建了 VSSBlock (`module/vss_block.py`)
- 对应架构图中的 **VSSM**
- 支持4方向扫描（`scan_direction='all'`）
- 基于 SS2D 实现，保持兼容性

### 2. ✅ 创建了 ChannelMambaBlock (`module/modules.py`)
- 对应架构图中的 **LFM**
- 通道维度的 Mamba，比注意力机制更轻量
- 已实现，可直接使用

### 3. ✅ 创建了 MineGateFusion (`basicsr/archs/minegate_fusion.py`)
- 实现双分支自适应融合
- 物理先验驱动：极暗信全局，局部强光信局部

### 4. ✅ 修改了 VSSLocalBlock (`basicsr/archs/vmamba.py`)
- 添加了参数开关，支持新旧模块切换
- 默认使用原始模块，可通过参数启用新模块

## 使用方法

### 方法1：在创建 VSSLocalBlock 时启用新模块

```python
from basicsr.archs.vmamba import VSSLocalBlock

# 启用所有新模块
block = VSSLocalBlock(
    hidden_dim=32,
    use_vssblock=True,        # 使用 VSSBlock 替换 SS2D
    use_channel_mamba=True,   # 使用 ChannelMambaBlock 替换 LocalFeatureModule
    use_minegate=True,        # 使用 MineGate 融合（已默认启用）
)

# 或者只启用部分
block = VSSLocalBlock(
    hidden_dim=32,
    use_channel_mamba=True,   # 只替换 LFM
)
```

### 方法2：修改 LIED 架构启用新模块

在 `basicsr/archs/lied_arch.py` 中：

```python
# 在 LIED.__init__ 中
self.encoder_level1 = nn.Sequential(*[
    VSSLocalBlock(
        hidden_dim=dim, 
        use_vssblock=True,        # 启用 VSSBlock
        use_channel_mamba=True,   # 启用 ChannelMambaBlock
        use_minegate=True,        # 启用 MineGate（默认已启用）
    ) 
    for i in range(num_blocks[0])
])
```

### 方法3：通过配置文件控制

可以在训练配置中添加参数：

```yaml
MODEL:
  USE_VSSBLOCK: true
  USE_CHANNEL_MAMBA: true
  USE_MINEGATE: true
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_vssblock` | bool | False | 使用 VSSBlock 替换 SS2D（VSSM） |
| `use_channel_mamba` | bool | False | 使用 ChannelMambaBlock 替换 LocalFeatureModule（LFM） |
| `use_minegate` | bool | True | 使用 MineGate 自适应融合（已默认启用） |
| `use_depthwise_ffn` | bool | False | 使用 DepthwiseConv2d 优化 FFN（暂未实现） |

## 推荐替换顺序

### 阶段1：替换 LFM（最简单，效果明显）
```python
VSSLocalBlock(hidden_dim=32, use_channel_mamba=True)
```

### 阶段2：替换 VSSM（需要验证兼容性）
```python
VSSLocalBlock(
    hidden_dim=32, 
    use_vssblock=True,
    use_channel_mamba=True,
)
```

### 阶段3：完整替换（所有新模块）
```python
VSSLocalBlock(
    hidden_dim=32,
    use_vssblock=True,
    use_channel_mamba=True,
    use_minegate=True,  # 已默认启用
)
```

## 预期效果

### 参数量对比
- **VSSBlock vs SS2D**: 基本相同（都基于 SS2D）
- **ChannelMambaBlock vs LocalFeatureModule**: 可能减少 10-20% 参数
- **MineGate vs Concat+1×1**: 增加约 50 个参数（可忽略）

### 性能预期
- **ChannelMambaBlock**: 通道建模更直接，可能提升局部特征提取
- **VSSBlock**: 保持全局建模能力，兼容性更好
- **MineGate**: 自适应融合，在极暗/局部强光场景下效果更明显

## 测试建议

1. **先测试 ChannelMambaBlock**
   ```python
   # 只替换 LFM
   block = VSSLocalBlock(hidden_dim=32, use_channel_mamba=True)
   ```

2. **再测试 VSSBlock**
   ```python
   # 替换 VSSM
   block = VSSLocalBlock(hidden_dim=32, use_vssblock=True)
   ```

3. **最后测试组合**
   ```python
   # 全部替换
   block = VSSLocalBlock(
       hidden_dim=32,
       use_vssblock=True,
       use_channel_mamba=True,
   )
   ```

## 注意事项

1. **VSSBlock 需要 SS2D**
   - 确保 `basicsr.archs.vmamba.SS2D` 可用
   - 如果不可用，会回退到原始 SS2D

2. **ChannelMambaBlock 已实现**
   - 在 `module/modules.py` 中
   - 需要确保导入路径正确

3. **MineGate 已默认启用**
   - 如果导入失败，会回退到原始融合方式
   - 建议保持启用状态

4. **向后兼容**
   - 默认参数保持原有行为
   - 只有显式启用新模块才会替换

## 故障排查

### 问题1：导入错误
```
Warning: New modules (VSSBlock, ChannelMambaBlock) not found
```
**解决**：检查 `module/vss_block.py` 和 `module/modules.py` 是否存在

### 问题2：VSSBlock 不工作
**解决**：确保 `SS2D` 可用，检查 `basicsr.archs.vmamba` 导入

### 问题3：ChannelMambaBlock 报错
**解决**：检查 `module/modules.py` 中的实现，确保 `SS2D` 可用




