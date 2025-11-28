"""
模块4: MineGate - Physics-Informed Dynamic Illumination Selection
独立模块文件
论文核心创新点：可解释的自适应门控机制
"""

import torch
import torch.nn as nn


class MineGate(nn.Module):
    """
    模块4: MineGate - Physics-Informed Dynamic Illumination Selection
    论文核心创新点：可解释的自适应门控机制
    
    硬逻辑（物理驱动）：
    - 全局极暗（mean≈0, std小, dark_ratio大）→ α→1（信全局 Illumination Branch）
    - 局部矿灯强光（std极大）→ α→0（信局部 Reflectance Branch）
    
    软残差（学习微调）：
    - 轻量MLP微调 (±0.3)，防止硬逻辑过于死板
    
    实现"极暗信全局，矿灯强光信局部"的自适应策略
    """
    def __init__(self):
        super().__init__()
        # 软学习残差：轻量MLP微调 (±0.3)
        # 输入：mean, std, dark_ratio
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),  # 轻量级
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # 输出范围 [-1, 1]，后续会缩放到 [-0.3, 0.3]
        )
    
    def forward(self, E0: torch.Tensor, F_illum: torch.Tensor, F_refl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E0: (B, 1, H, W) 光照图
            F_illum: (B, 64, H, W) 全局光照分支特征
            F_refl: (B, 64, H, W) 局部反射率分支特征
        Returns:
            F_fused: (B, 64, H, W) 融合后的特征
        """
        B = E0.shape[0]
        
        # 提取统计量
        mean = E0.mean(dim=[1, 2, 3])  # (B,) 全局亮度
        std = E0.std(dim=[1, 2, 3])    # (B,) 光照不均匀度（矿灯强光时 std 极大）
        dark_ratio = (E0 < 0.05).float().mean(dim=[1, 2, 3])  # (B,) 极暗区域比例（<5%亮度）
        
        # ========== 可解释的硬逻辑（物理驱动）==========
        # std大或暗区少 → 全局权重小（信局部）
        # 公式：global_weight = sigmoid(-15 * (std - 0.3)) * (1 - dark_ratio + 0.5)
        # - 当 std > 0.3（光照不均匀，有强光）→ sigmoid 值小 → global_weight 小
        # - 当 dark_ratio 大（极暗区域多）→ (1 - dark_ratio + 0.5) 小 → global_weight 小
        # - 当全局极暗（std小，dark_ratio大）→ global_weight 大（信全局）
        global_weight = torch.sigmoid(-15 * (std - 0.3)) * (1 - dark_ratio + 0.5)
        global_weight = global_weight.clamp(0.0, 1.0)
        
        # ========== 软学习残差（防止硬逻辑太死板）==========
        stats = torch.stack([mean, std, dark_ratio], dim=1)  # (B, 3)
        residual = self.mlp(stats)  # (B, 1) ∈ [-1, 1]
        residual = residual * 0.3  # 缩放到 [-0.3, 0.3]
        
        # 融合硬逻辑和软残差
        alpha = global_weight.unsqueeze(1) + residual  # (B, 1)
        alpha = alpha.clamp(0.0, 1.0)
        alpha = alpha.view(B, 1, 1, 1)  # (B, 1, 1, 1)
        
        # 特征融合
        F_fused = alpha * F_illum + (1 - alpha) * F_refl
        
        return F_fused

