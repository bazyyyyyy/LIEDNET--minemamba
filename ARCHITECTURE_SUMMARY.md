# LIEDNet 网络架构总结

## 5 个核心模块

### 1. Overlap Patch Embedding（重叠图像块嵌入）
- **3×3 卷积，stride=1，padding=1**（Restormer 风格，不降采样）
- 输入：(B, 3, H, W) → 输出：(B, 32, H, W)
- 保持空间分辨率不变

### 2. Encoder（编码器）
- **3 级层次结构**（Restormer 风格）
- **Level 1**: 32 通道，1 个 VSSLocalBlock
- **Level 2**: 64 通道，2 个 VSSLocalBlock（通过 Downsample: PixelUnshuffle）
- **Level 3**: 128 通道，4 个 VSSLocalBlock（通过 Downsample: PixelUnshuffle）
- 空间分辨率：H×W → H/2×W/2 → H/4×W/4

### 3. Bottleneck（瓶颈层）
- **1 个 VSSLocalBlock**（128 通道）
- 特征融合：F₃ → VSSLocalBlock → F₃'
- 空间分辨率：H/4×W/4

### 4. Decoder（解码器）
- **3 级上采样结构**（带 Skip Connections）
- **Level 2**: 64 通道，2 个 VSSLocalBlock
  - Upsample (PixelShuffle) + Skip Connection (Level 2) + 1×1 Conv 融合
- **Level 1**: 64 通道，1 个 VSSLocalBlock
  - Upsample (PixelShuffle) + Skip Connection (Level 1)
- **Refinement**: 64 通道，4 个 VSSLocalBlock
- 空间分辨率：H/4×W/4 → H/2×W/2 → H×W

### 5. Output Layer（输出层）
- **3×3 卷积**：64 通道 → 3 通道
- **残差连接**：enhanced = input + residual

## VSSLocalBlock 核心组件

### SS2D（State Space Model，四方向扫描）
- **四方向扫描**：H（水平）、V（垂直）、forward-diagonal（前向对角）、backward-diagonal（后向对角）
- **d_state=16**，**ssm_ratio=2.0**
- **3×3 深度可分离卷积**（d_conv=3）
- **激活函数**：SiLU

### LocalFeatureModule（局部特征模块）
- **通道注意力**：AdaptiveAvgPool2d → 1×1 Conv (dim → dim/2 → dim) → Sigmoid
- **空间注意力**：3×3 Conv (dim → 1) → Sigmoid
- **双重注意力融合**：x * channel_att * spatial_att

### DGDFeedForward（双门控深度可分离前馈网络）
- **三分支结构**：
  - 分支1：5×5 深度可分离卷积 + ReLU
  - 分支2：3×3 深度可分离卷积 + ReLU
  - 分支3：3×3 深度可分离卷积 + GELU
- **融合**：分支1+2 融合后与分支3 相乘
- **扩展因子**：2.66

## 关键特性

- **激活函数**：
  - SS2D: SiLU
  - LocalFeatureModule: ReLU, Sigmoid
  - DGDFeedForward: ReLU, GELU
- **归一化**：LayerNorm（pre/post 每个 VSSLocalBlock）
- **下采样**：PixelUnshuffle（2×2 → 通道数减半）
- **上采样**：PixelShuffle（2×2 → 通道数减半）
- **Skip Connections**：Encoder Level 1/2 → Decoder Level 1/2
- **动态分辨率**：支持任意 H×W（通过 padding 确保 %64==0）
- **参数量**：~5.26M（根据训练日志）
- **FLOPs**：~74.08 GMac（128×128 输入，根据训练日志）

## 架构流程

```
Input (B, 3, H, W)
    ↓
OverlapPatchEmbed (3×3, s=1) → (B, 32, H, W)
    ↓
Encoder Level 1: 1× VSSLocalBlock → (B, 32, H, W)
    ↓ Downsample (PixelUnshuffle)
Encoder Level 2: 2× VSSLocalBlock → (B, 64, H/2, W/2)
    ↓ Downsample (PixelUnshuffle)
Encoder Level 3: 4× VSSLocalBlock → (B, 128, H/4, W/4)
    ↓
Bottleneck: 1× VSSLocalBlock → (B, 128, H/4, W/4)
    ↓
Decoder Level 2: Upsample + Skip(L2) + 2× VSSLocalBlock → (B, 64, H/2, W/2)
    ↓
Decoder Level 1: Upsample + Skip(L1) + 1× VSSLocalBlock → (B, 64, H, W)
    ↓
Refinement: 4× VSSLocalBlock → (B, 64, H, W)
    ↓
Output: 3×3 Conv → (B, 3, H, W)
    ↓
Residual: input + output → Enhanced Image
```

## 训练特性

- **混合精度训练（AMP）**：启用
- **边缘损失**：Sobel 梯度 L1 损失（权重 0.05）
- **Batch Size**：4（优化内存）
- **学习率调度**：Warmup (3 epochs) + CosineAnnealing
- **优化器**：Adam (lr=1e-4, betas=(0.9, 0.999))




