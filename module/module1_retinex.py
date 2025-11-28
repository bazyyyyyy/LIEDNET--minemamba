"""
模块1: Initial Retinex Decomposition
独立模块文件
"""

import torch
import torch.nn as nn
from typing import Tuple


class InitialRetinexDecomposition(nn.Module):
    """
    模块1: Initial Retinex Decomposition
    使用2层1x1卷积进行轻量级分解
    输入: 低光图像 (B, 3, H, W)
    输出: 粗光照 E0 (B, 1, H, W) + 粗反射率 R0 (B, 3, H, W)
    """
    def __init__(self):
        super().__init__()
        # 第一层：提取光照信息
        self.illum_conv1 = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.illum_conv2 = nn.Conv2d(16, 1, kernel_size=1, bias=False)
        self.illum_act = nn.SiLU()
        
        # 第二层：提取反射率信息
        self.refl_conv1 = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.refl_conv2 = nn.Conv2d(16, 3, kernel_size=1, bias=False)
        self.refl_act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 低光图像
        Returns:
            E0: (B, 1, H, W) 粗光照
            R0: (B, 3, H, W) 粗反射率
        """
        # 光照分支
        E0 = self.illum_conv1(x)
        E0 = self.illum_act(E0)
        E0 = self.illum_conv2(E0)
        E0 = torch.sigmoid(E0)  # 确保光照在 [0, 1]
        
        # 反射率分支
        R0 = self.refl_conv1(x)
        R0 = self.refl_act(R0)
        R0 = self.refl_conv2(R0)
        R0 = torch.sigmoid(R0)  # 确保反射率在 [0, 1]
        
        return E0, R0

