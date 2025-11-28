# 5个模块分析：挑选最重要的2个替换 VSSLocalBlock 子模块

## 结论：最重要的2个模块

### ✅ **module3_reflectance 中的两个核心模块**

1. **`MambaBlock`** → 替换 **SS2D (SSM模块)**
2. **`ChannelMambaBlock`** → 替换 **LocalFeatureModule (LFM模块)**

## 为什么选这两个？

### 当前 VSSLocalBlock 结构：
```
VSSLocalBlock:
  ├─ SS2D (全局分支，4方向扫描，SelectiveScan)
  ├─ LocalFeatureModule (局部分支，通道+空间注意力)
  └─ DGDFeedForward (前馈网络)
```

### module3_reflectance 中的设计：
```python
# 每级处理流程：
DepthwiseConv2d → SiLU → MambaBlock → ChannelMambaBlock
```

**关键洞察**：
- `MambaBlock` 是**轻量化的 Mamba 实现**，可能比 SS2D 更高效
- `ChannelMambaBlock` 是**通道维度的 Mamba**，比 LocalFeatureModule 的注意力机制更适合局部特征建模
- 两者组合：**空间 Mamba (MambaBlock) + 通道 Mamba (ChannelMambaBlock)** = 更轻量且可能更有效的全局+局部建模

## 其他模块为什么不合适？

| 模块 | 作用 | 是否适合替换 |
|------|------|------------|
| module1_retinex | 输入级 Retinex 分解 | ❌ 输入级模块，不是 block 内组件 |
| module2_illumination | 完整的分支网络（4级 U-Net） | ❌ 太大，是整个分支 |
| module3_reflectance | **MambaBlock + ChannelMambaBlock** | ✅ **核心模块** |
| module4_minegate | 分支融合门控 | ❌ 跨分支融合，不是 block 内模块 |
| module5_decoder | U-Net 解码器 | ❌ 解码器级模块 |

## 替换方案

### 方案1：直接替换（如果 MambaBlock/ChannelMambaBlock 已实现）

```python
# 在 VSSLocalBlock 中：
# 原来：
self.op = SS2D(...)  # 替换为
self.op = MambaBlock(d_model=hidden_dim)

self.local = LocalFeatureModule(...)  # 替换为
self.local = ChannelMambaBlock(d_model=hidden_dim)
```

### 方案2：需要先实现这两个模块

如果 `module/modules.py` 不存在，需要基于以下接口实现：

**MambaBlock 接口**：
- 输入：`(B, C, H, W)` 或 `(B, H, W, C)`
- 输出：同尺寸
- 功能：轻量化的空间 Mamba（可能比 SS2D 的 4 方向扫描更简单）

**ChannelMambaBlock 接口**：
- 输入：`(B, C, H, W)`
- 输出：同尺寸
- 功能：通道维度的 Mamba（将通道作为序列处理）

## 预期效果

### 参数量对比（假设）：
- **SS2D**: ~O(C²) (4方向扫描 + SelectiveScan)
- **MambaBlock**: ~O(C) (简化的 Mamba，可能单方向或2方向)
- **LocalFeatureModule**: ~O(C²) (通道注意力 + 空间注意力)
- **ChannelMambaBlock**: ~O(C) (通道序列 Mamba)

**预期减参**：可能减少 20-40% 的 block 内参数

### 性能预期：
- **MambaBlock** 可能比 SS2D 更专注（单/双方向 vs 4方向），计算更高效
- **ChannelMambaBlock** 的通道建模可能比注意力机制更适合局部特征
- **组合效果**：空间 Mamba + 通道 Mamba = 更轻量且可能更有效的特征提取

## 下一步行动

1. **检查 `module/modules.py` 是否存在**，如果不存在需要实现
2. **实现 MambaBlock 和 ChannelMambaBlock**（基于现有 SS2D 和 LocalFeatureModule 的设计思路）
3. **修改 VSSLocalBlock**，替换 SS2D → MambaBlock，LocalFeatureModule → ChannelMambaBlock
4. **测试参数量和性能**




