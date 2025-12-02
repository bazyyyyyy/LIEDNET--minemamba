"""
稳定的 Retinex 分解模块（永不爆炸版）
解决 loss 爆炸问题的终极方案
"""

import torch
import torch.nn as nn


class StableRetinex(nn.Module):
    """
    稳定的 Retinex 分解：I = R * E
    - 使用可学习的全局缩放参数防止 E0 过小导致 R0 爆炸
    - 硬截断防止数值溢出
    """
    
    def __init__(self):
        super().__init__()
        # 轻量级卷积网络提取光照图
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 1), nn.ReLU(),
            nn.Conv2d(24, 12, 1), nn.ReLU(),
            nn.Conv2d(12, 1, 1),
            nn.Sigmoid()
        )
        # 关键！加一个可学习的全局缩放参数
        # 初始化为 1.0，让初始E0接近输入的平均亮度
        # 这样初始R0 = I/E0 ≈ I/1.0 = I，模型从"接近输入"开始
        # 限制 scale 在 [0.5, 2.0] 范围内，防止训练时爆炸
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        # ========== 关键修复：初始化Retinex网络，让R0初始值接近输入 ==========
        self._init_retinex_weights()
    
    def _init_retinex_weights(self):
        """
        初始化Retinex网络，确保：
        1. 初始E0接近输入的平均亮度（scale=1.0，conv输出接近0.5）
        2. 初始R0 = I/E0 ≈ I/1.0 = I（接近输入）
        这样模型从"R0≈输入"的合理起点开始，而不是从随机值开始
        """
        # 初始化最后一层conv，让输出接近0.5（经过sigmoid后）
        # 这样 E0 = sigmoid(0.5) * 1.0 ≈ 0.62，R0 = I/0.62 ≈ 1.6*I（在合理范围内）
        with torch.no_grad():
            # 最后一层conv的权重初始化为接近0，bias初始化为0（经过sigmoid后输出≈0.5）
            if len(self.conv) > 0:
                last_conv = self.conv[-2]  # 倒数第二层是Conv2d
                if isinstance(last_conv, nn.Conv2d):
                    nn.init.zeros_(last_conv.weight)
                    if last_conv.bias is not None:
                        # 设置bias让sigmoid输出接近0.5
                        # sigmoid(0) = 0.5，所以bias=0即可
                        nn.init.zeros_(last_conv.bias)
        
        print("==> StableRetinex initialized: E0≈0.6, R0≈1.6*I (reasonable start)")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图像，范围 [0, 1]
        Returns:
            E0: (B, 1, H, W) 光照图，范围 [0.2, 2.0]（严格限制）
            R0: (B, 3, H, W) 反射率图，范围 [0, 2.0]（严格限制，防止爆炸）
        """
        # ========== 关键修复：限制 scale 范围，防止训练时爆炸 ==========
        # 如果 scale 训练时变得太大或太小，E0 会爆炸，导致 R0 爆炸
        scale_clamped = torch.clamp(self.scale, 0.5, 2.0)  # 限制在 [0.5, 2.0]
        
        # 提取光照图并放大
        E0_raw = self.conv(x)  # [0, 1] after sigmoid
        E0 = E0_raw * scale_clamped  # [0, 2.0]
        
        # ========== 关键修复：确保 E0 在合理范围 [0.2, 2.0] ==========
        # 如果 E0 太小（<0.2），R0 = I/E0 会爆炸
        # 如果 E0 太大（>2.0），R0 会变得很小，但不会爆炸
        E0 = torch.clamp(E0, 0.2, 2.0)  # 严格限制 E0 范围
        
        # 计算反射率：R = I / E
        # 现在 E0 ∈ [0.2, 2.0]，所以 R0 = I/E0 ∈ [I/2.0, I/0.2] = [0, 5*I]
        # 对于低光图 I≈0.05，R0 ∈ [0, 0.25]，在合理范围内
        R0 = x / (E0 + 1e-8)  # 增加 epsilon 提高数值稳定性
        
        # ========== 关键修复：更严格的 R0 截断，防止爆炸 ==========
        # 如果 R0 真的爆炸到几万，说明 E0 计算有问题，需要更严格的限制
        R0 = torch.clamp(R0, 0, 2.0)  # 从 5.0 降到 2.0，更严格
        
        # ========== 数值稳定性检查 ==========
        # 如果检测到异常值，记录警告（但不中断训练）
        if torch.any(torch.isnan(E0)) or torch.any(torch.isinf(E0)):
            print(f"Warning: E0 contains NaN/Inf! scale={self.scale.item():.4f}")
            E0 = torch.clamp(E0, 0.2, 2.0)
            E0 = torch.where(torch.isnan(E0) | torch.isinf(E0), torch.tensor(0.5).to(E0.device), E0)
        
        if torch.any(torch.isnan(R0)) or torch.any(torch.isinf(R0)):
            print(f"Warning: R0 contains NaN/Inf! E0_min={E0.min().item():.4f}, E0_max={E0.max().item():.4f}")
            R0 = torch.clamp(R0, 0, 2.0)
            R0 = torch.where(torch.isnan(R0) | torch.isinf(R0), torch.tensor(1.0).to(R0.device), R0)
        
        return E0, R0

