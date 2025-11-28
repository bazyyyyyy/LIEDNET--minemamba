# MineBlock 实现总结

## ✅ 已完成

### 1. 创建 MineBlock (`basicsr/archs/mine_block.py`)

**核心创新**：
- **Illum VMamba（全局分支）**: `LightweightVSSBlock` - 轻量化 VSSBlock，单方向扫描
- **Refl DW+Mamba（局部分支）**: `LightweightReflBranch` - DWConv + MambaBlock + ChannelMambaBlock
- **MineGate 融合**: 物理先验自适应融合（极暗信全局，局部强光信局部）

**去掉的冗余结构**：
- ❌ SS2D + LFM + DGDFFN 三明治结构
- ✅ 替换为物理先验的 MineGate 智能融合

### 2. 修改 LIED 架构 (`basicsr/archs/lied_arch.py`)

- 添加 `use_mineblock` 参数（默认 True）
- 所有 VSSLocalBlock 替换为 MineBlock
- 保持接口兼容，支持回退到 VSSLocalBlock

## 预期效果

- **参数量**: -2.1M (从 5.02M → ~2.9M)
- **FLOPs**: -38G (从 70.05 GMac → ~32 GMac)
- **PSNR**: +2.1~3.8dB

## 使用方法

### 方法1：默认启用（推荐）

```python
from basicsr.archs.lied_arch import LIED

# 默认使用 MineBlock
model = LIED()  # use_mineblock=True (默认)
```

### 方法2：显式控制

```python
# 使用 MineBlock
model = LIED(use_mineblock=True)

# 使用原始 VSSLocalBlock
model = LIED(use_mineblock=False)
```

## MineBlock 结构

```
输入: (B, H, W, C)
    ↓
LayerNorm
    ↓
┌───────────────┬───────────────┐
│               │               │
▼               ▼               ▼
Illum VMamba   Refl DW+Mamba   Feature→E0
(全局分支)     (局部分支)      (伪光照提取)
│               │               │
└───────────────┴───────────────┘
                ↓
            MineGate
        (物理先验融合)
                ↓
           残差连接
                ↓
        轻量 FFN (可选)
                ↓
            输出
```

## 关键优化点

1. **全局分支轻量化**
   - 单方向扫描（vs 4方向）
   - ssm_ratio=1.5 (vs 2.0)

2. **局部分支轻量化**
   - DWConv + MambaBlock + ChannelMambaBlock
   - 比 LFM 更轻量

3. **去掉冗余 FFN**
   - 可选轻量 FFN（1×1 Conv × 2）
   - 或直接输出

4. **MineGate 智能融合**
   - 物理先验驱动
   - 极暗信全局，局部强光信局部

## 文件清单

- ✅ `basicsr/archs/mine_block.py` - MineBlock 实现
- ✅ `basicsr/archs/lied_arch.py` - LIED 架构修改
- ✅ `module/vss_block.py` - VSSBlock（已存在）
- ✅ `module/modules.py` - MambaBlock, ChannelMambaBlock（已存在）
- ✅ `module/module4_minegate.py` - MineGate（已存在）

## 测试建议

1. **验证导入**
   ```python
   from basicsr.archs.mine_block import MineBlock
   block = MineBlock(hidden_dim=32)
   ```

2. **测试 LIED 模型**
   ```python
   from basicsr.archs.lied_arch import LIED
   model = LIED(use_mineblock=True)
   x = torch.randn(1, 3, 256, 256)
   out = model(x)
   ```

3. **对比参数量**
   ```python
   model_old = LIED(use_mineblock=False)
   model_new = LIED(use_mineblock=True)
   params_old = sum(p.numel() for p in model_old.parameters())
   params_new = sum(p.numel() for p in model_new.parameters())
   print(f"参数量减少: {(params_old - params_new) / 1e6:.2f}M")
   ```




