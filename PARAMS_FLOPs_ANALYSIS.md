# 参数量和 FLOPs 分析

## 当前模型统计
- **参数量**: 5.02M
- **FLOPs**: 70.05 GMac (输入: 256×256)

## 主要影响模块和代码文件

### 1. VSSLocalBlock（核心模块，贡献最大）

**代码文件**: `basicsr/archs/vmamba.py` (Line 1292-1498)

**参数量贡献**: ~60-70% (约 3.0-3.5M)
**FLOPs 贡献**: ~65-75% (约 45-52 GMac)

#### VSSLocalBlock 内部组件：

##### 1.1 SS2D (VSSM) - 全局分支
**位置**: `basicsr/archs/vmamba.py` (Line 548-1107)
**参数量**: ~40-50% of VSSLocalBlock
**FLOPs**: ~50-60% of VSSLocalBlock

**关键参数**:
- `ssm_ratio=2.0` → d_inner = 2 × d_model（**参数量翻倍**）
- `d_state=16` → SelectiveScan 状态维度
- `d_conv=3` → 3×3 深度可分离卷积
- 4方向扫描（H/V/forward-diag/backward-diag）

**主要参数来源**:
```python
# Line 750-776: in_proj, x_proj, dt_projs, A_logs, Ds, out_proj
self.in_proj = Linear(d_model, d_inner * 2)  # 2倍扩展
self.x_proj = [Linear(d_inner, dt_rank + d_state * 2) for _ in range(4)]  # 4方向
self.dt_projs_weight = (K, d_inner, dt_rank)  # K=4
self.A_logs = (K * d_inner, d_state)  # 状态矩阵
self.out_proj = Linear(d_inner, d_model)
```

**FLOPs 主要来源**:
- SelectiveScan: `9 * B * L * D * N` (Line 198)
- 4方向扫描 × 4 = 4倍计算量

##### 1.2 LocalFeatureModule (LFM) - 局部分支
**位置**: `basicsr/archs/vmamba.py` (Line 1212-1241)
**参数量**: ~10-15% of VSSLocalBlock
**FLOPs**: ~5-10% of VSSLocalBlock

**关键参数**:
```python
# Line 1219-1230: 通道注意力 + 空间注意力
self.channel_weight = Sequential(
    Conv2d(dim, dim//2, 1),  # 降维
    ReLU,
    Conv2d(dim//2, dim, 1),  # 恢复
    Sigmoid
)
self.spatial_weight = Sequential(
    Conv2d(dim, 1, 3, padding=1),  # 空间注意力
    Sigmoid
)
```

##### 1.3 DGDFeedForward (FFN) - 前馈网络
**位置**: `basicsr/archs/vmamba.py` (Line 1260-1289)
**参数量**: ~30-40% of VSSLocalBlock
**FLOPs**: ~25-35% of VSSLocalBlock

**关键参数**:
- `ffn_expansion_factor=2.66` → hidden_features = 2.66 × dim（**参数量大**）

**主要参数来源**:
```python
# Line 1265-1277: 三分支深度可分离卷积
hidden_features = int(dim * 2.66)  # 扩展因子大
self.project_in1 = Conv2d(dim, hidden_features * 2, 1)  # 2倍
self.dwconv1 = Conv2d(hidden_features, hidden_features, 5, groups=hidden_features)
self.dwconv2 = Conv2d(hidden_features, hidden_features, 3, groups=hidden_features)
self.dwconv3 = Conv2d(hidden_features, hidden_features, 3, groups=hidden_features)
self.fusion = Conv2d(hidden_features * 2, hidden_features, 1)
self.project_out = Conv2d(hidden_features, dim, 1)
```

**FLOPs 主要来源**:
- 3个 DWConv (5×5, 3×3, 3×3) × hidden_features
- hidden_features = 2.66 × dim，计算量大

##### 1.4 MineGate 融合（新增）
**位置**: `basicsr/archs/minegate_fusion.py`
**参数量**: ~0.1% (约 50 个参数，可忽略)
**FLOPs**: ~0.1% (可忽略)

### 2. LIED 架构层级分布

**代码文件**: `basicsr/archs/lied_arch.py`

#### 2.1 Encoder Level 3（参数量最大）
**位置**: Line 71
**配置**: 4个 VSSLocalBlock，通道数 128 (dim×2²)
**参数量贡献**: ~25-30% (约 1.25-1.5M)
**FLOPs 贡献**: ~20-25% (约 14-17 GMac)

**原因**:
- 通道数最大：128 (vs Level 1的32, Level 2的64)
- Block数量最多：4个
- 空间分辨率：H/4 × W/4，但通道数大，总计算量仍大

#### 2.2 Refinement（参数量第二）
**位置**: Line 85
**配置**: 4个 VSSLocalBlock，通道数 64 (dim×2¹)
**参数量贡献**: ~15-20% (约 0.75-1.0M)
**FLOPs 贡献**: ~15-20% (约 10-14 GMac)

**原因**:
- Block数量多：4个
- 空间分辨率：H × W（全分辨率）

#### 2.3 Encoder Level 2
**位置**: Line 68
**配置**: 2个 VSSLocalBlock，通道数 64
**参数量贡献**: ~8-10% (约 0.4-0.5M)
**FLOPs 贡献**: ~8-10% (约 5-7 GMac)

#### 2.4 Encoder Level 1
**位置**: Line 65
**配置**: 1个 VSSLocalBlock，通道数 32
**参数量贡献**: ~3-5% (约 0.15-0.25M)
**FLOPs 贡献**: ~5-8% (约 3-5 GMac)

#### 2.5 Decoder Levels
**位置**: Line 80, 83
**配置**: 2个 + 1个 VSSLocalBlock，通道数 64
**参数量贡献**: ~8-12% (约 0.4-0.6M)
**FLOPs 贡献**: ~10-15% (约 7-10 GMac)

#### 2.6 Bottleneck
**位置**: Line 75
**配置**: 1个 VSSLocalBlock，通道数 128
**参数量贡献**: ~6-8% (约 0.3-0.4M)
**FLOPs 贡献**: ~5-8% (约 3-5 GMac)

#### 2.7 其他模块（PatchEmbed, Downsample, Upsample, Output）
**位置**: Line 14-46, 87
**参数量贡献**: ~5-8% (约 0.25-0.4M)
**FLOPs 贡献**: ~3-5% (约 2-3 GMac)

## 参数量详细分解（估算）

| 模块 | 参数量 (M) | 占比 | 主要来源 |
|------|-----------|------|---------|
| **VSSLocalBlock (总计)** | **~3.5** | **70%** | SS2D + DGDFFN + LFM |
| ├─ SS2D (VSSM) | ~1.4 | 28% | ssm_ratio=2.0, 4方向扫描 |
| ├─ DGDFFN | ~1.2 | 24% | ffn_expansion_factor=2.66 |
| ├─ LFM | ~0.5 | 10% | 通道+空间注意力 |
| └─ 其他 | ~0.4 | 8% | LayerNorm, 融合层等 |
| **Encoder Level 3** | **~1.3** | **26%** | 4 blocks × 128 channels |
| **Refinement** | **~0.9** | **18%** | 4 blocks × 64 channels |
| **Encoder Level 2** | **~0.4** | **8%** | 2 blocks × 64 channels |
| **Decoder Levels** | **~0.5** | **10%** | 3 blocks × 64 channels |
| **其他** | **~0.4** | **8%** | PatchEmbed, Down/Upsample, Output |
| **总计** | **~5.0** | **100%** | |

## FLOPs 详细分解（估算，输入256×256）

| 模块 | FLOPs (GMac) | 占比 | 主要来源 |
|------|-------------|------|---------|
| **VSSLocalBlock (总计)** | **~48** | **68%** | SS2D + DGDFFN + LFM |
| ├─ SS2D (VSSM) | ~28 | 40% | SelectiveScan × 4方向 |
| ├─ DGDFFN | ~15 | 21% | 3分支 DWConv × 2.66扩展 |
| ├─ LFM | ~3 | 4% | 通道+空间注意力 |
| └─ 其他 | ~2 | 3% | LayerNorm, 融合等 |
| **Encoder Level 3** | **~12** | **17%** | 4 blocks, 128ch, H/4×W/4 |
| **Refinement** | **~8** | **11%** | 4 blocks, 64ch, H×W |
| **Encoder Level 2** | **~4** | **6%** | 2 blocks, 64ch, H/2×W/2 |
| **Decoder Levels** | **~5** | **7%** | 3 blocks, 64ch, H/2×W/2, H×W |
| **其他** | **~2** | **3%** | PatchEmbed, Down/Upsample, Output |
| **总计** | **~70** | **100%** | |

## 优化建议（减少参数量和FLOPs）

### 1. 减少 SS2D 参数量（影响最大）
**文件**: `basicsr/archs/vmamba.py` (Line 548-1107)

**方法**:
- 降低 `ssm_ratio` 从 2.0 → 1.5（减少 25% 参数）
- 使用单方向或双方向扫描（减少 50-75% FLOPs）
- 降低 `d_state` 从 16 → 12（减少约 25% 参数）

**预期效果**: 参数量 -15~20%, FLOPs -20~30%

### 2. 减少 DGDFFN 参数量
**文件**: `basicsr/archs/vmamba.py` (Line 1260-1289)

**方法**:
- 降低 `ffn_expansion_factor` 从 2.66 → 2.0（减少 25% 参数）
- 简化三分支为双分支（减少 33% 参数和FLOPs）

**预期效果**: 参数量 -10~15%, FLOPs -10~15%

### 3. 减少 Block 数量
**文件**: `basicsr/archs/lied_arch.py` (Line 65, 68, 71, 80, 83, 85)

**方法**:
- Encoder Level 3: 4 → 3 blocks
- Refinement: 4 → 3 blocks

**预期效果**: 参数量 -10~15%, FLOPs -10~15%

### 4. 使用 ChannelMambaBlock 替换 LFM
**文件**: `module/modules.py` (ChannelMambaBlock)

**方法**:
- 使用 `use_channel_mamba=True` 启用
- ChannelMambaBlock 比 LocalFeatureModule 更轻量

**预期效果**: 参数量 -5~10%, FLOPs -3~5%

### 5. 降低通道数
**文件**: `basicsr/archs/lied_arch.py` (Line 53-58)

**方法**:
- 降低 `dim` 从 32 → 28 或 24

**预期效果**: 参数量 -20~30%, FLOPs -20~30%（但可能影响性能）

## 关键代码位置总结

### 参数量主要来源（按重要性排序）

1. **SS2D (VSSM)** - `basicsr/archs/vmamba.py:548-1107`
   - `ssm_ratio=2.0` (Line 1318)
   - `d_state=16` (Line 1300)
   - 4方向扫描

2. **DGDFFN** - `basicsr/archs/vmamba.py:1260-1289`
   - `ffn_expansion_factor=2.66` (Line 1405)

3. **Encoder Level 3** - `basicsr/archs/lied_arch.py:71`
   - 4个 blocks × 128 channels

4. **Refinement** - `basicsr/archs/lied_arch.py:85`
   - 4个 blocks × 64 channels

### FLOPs 主要来源（按重要性排序）

1. **SS2D SelectiveScan** - `basicsr/archs/vmamba.py:198`
   - `9 * B * L * D * N` × 4方向

2. **DGDFFN DWConv** - `basicsr/archs/vmamba.py:1269-1271`
   - 3个分支 × hidden_features (2.66×dim)

3. **Encoder Level 3** - `basicsr/archs/lied_arch.py:71`
   - 4 blocks × 128ch × H/4×W/4

4. **Refinement** - `basicsr/archs/lied_arch.py:85`
   - 4 blocks × 64ch × H×W

## 快速优化方案

如果要快速减少参数量和FLOPs，按优先级：

1. **降低 ssm_ratio**: 2.0 → 1.5（修改 `basicsr/archs/vmamba.py:1301`）
2. **降低 ffn_expansion_factor**: 2.66 → 2.0（修改 `basicsr/archs/vmamba.py:1405`）
3. **减少 Refinement blocks**: 4 → 3（修改 `basicsr/archs/lied_arch.py:58`）
4. **启用 ChannelMambaBlock**: `use_channel_mamba=True`

**预期总效果**: 参数量 -25~35%, FLOPs -25~35%




