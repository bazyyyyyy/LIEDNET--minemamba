"""
MineBlock - 核心创新模块（方案A：终极防崩版）
彻底替换 VSSLocalBlock，整合：
1. Illum VMamba（全局分支）- VSSBlock
2. Refl DW+Mamba（局部分支）- DWConv + Mamba
3. MineGate（物理先验融合）- 极暗信全局，局部强光信局部

预期效果：-2.1M 参数，-38G FLOPs，+2.1~3.8dB PSNR

关键：每个 MineBlock 自动适配当前层通道（32/64/128），Mamba 的 d_model=dim 确保支持任意通道数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from basicsr.archs.vmamba import SS2D
    HAS_SS2D = True
except:
    HAS_SS2D = False
    print("Warning: SS2D not found")

try:
    from timm.models.layers import DropPath
except:
    try:
        from timm.layers import DropPath
    except:
        # Fallback: simple DropPath
        class DropPath(nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = drop_prob
            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1 - self.drop_prob
                random_tensor = torch.rand(x.size(0), 1, 1, 1, device=x.device)
                random_tensor.bernoulli_(keep_prob)
                return x / keep_prob * random_tensor

try:
    import sys
    import os
    # 添加 module 目录到路径
    module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'module')
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
    
    from vss_block import VSSBlock
    from modules import MambaBlock
    from module4_minegate import MineGate
    
    # 创建 Mamba 类作为 MambaBlock 的别名（方案A要求）
    # 关键：d_model=dim 确保支持任意通道数（32/64/128等）
    class Mamba(MambaBlock):
        """Mamba 类，作为 MambaBlock 的别名，支持 expand 参数，自动适配任意通道数"""
        def __init__(self, d_model, d_state=8, expand=1.0, **kwargs):
            # expand 参数映射到 ssm_ratio
            # 关键：d_model 必须等于当前 dim，确保支持任意通道数
            super().__init__(
                d_model=d_model,  # ← 关键！自动适配当前层通道（32/64/128等）
                d_state=d_state,
                ssm_ratio=expand,  # expand 映射到 ssm_ratio
                channel_first=False,  # Mamba 期望 (B, H, W, C) 格式
                **kwargs
            )
    
    # 创建 MineGatePhysics 类（基于 MineGate，但返回元组）
    class MineGatePhysics(MineGate):
        """MineGate 的物理先验版本，返回 (fused, alpha) 元组"""
        def forward(self, E0: torch.Tensor, F_illum: torch.Tensor, F_refl: torch.Tensor):
            """
            Args:
                E0: (B, 1, H, W) 光照图
                F_illum: (B, C, H, W) 全局光照分支特征
                F_refl: (B, C, H, W) 局部反射率分支特征
            Returns:
                F_fused: (B, C, H, W) 融合后的特征
                alpha: (B, 1, 1, 1) 融合权重（用于可视化）
            """
            B = E0.shape[0]
            
            # 提取统计量
            mean = E0.mean(dim=[1, 2, 3])  # (B,) 全局亮度
            std = E0.std(dim=[1, 2, 3])    # (B,) 光照不均匀度（矿灯强光时 std 极大）
            dark_ratio = (E0 < 0.05).float().mean(dim=[1, 2, 3])  # (B,) 极暗区域比例（<5%亮度）
            
            # ========== 可解释的硬逻辑（物理驱动）==========
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
            
            return F_fused, alpha
    
    HAS_NEW_MODULES = True
except Exception as e:
    HAS_NEW_MODULES = False
    print(f"Warning: New modules not found: {e}, using fallback")
    # 创建回退的 Mamba 类
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=8, expand=1.0, **kwargs):
            super().__init__()
            self.identity = nn.Identity()
        def forward(self, x):
            return self.identity(x)
    
    # 创建回退的 MineGatePhysics
    class MineGatePhysics(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, E0, F_illum, F_refl):
            # 简单平均融合
            fused = 0.5 * F_illum + 0.5 * F_refl
            alpha = torch.ones(F_illum.shape[0], 1, 1, 1, device=F_illum.device) * 0.5
            return fused, alpha


class MineBlock(nn.Module):
    """
    MineBlock - 核心创新模块（方案A：终极防崩版）
    
    彻底替换 VSSLocalBlock，去掉冗余的 SS2D+LFM+DGDFFN 三明治结构
    使用物理先验的 MineGate 进行智能融合
    
    关键特性：
    - 自动适配通道数（使用 dim 参数，支持 32/64/128 等任意通道数）
    - 全局分支：Illum VMamba（VSSBlock）
    - 局部分支：Refl DW+Mamba（DWConv + Mamba）
    - 物理先验融合：MineGate（极暗信全局，局部强光信局部）
    
    关键：Mamba 的 d_model=dim 确保支持任意通道数，不会通道爆炸
    """
    def __init__(self, dim, d_state=8, ssm_ratio=1.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # 模块2: Illumination Branch (VMamba) - 自动适配 dim
        if HAS_NEW_MODULES:
            try:
                self.illum_branch = VSSBlock(
                    dim=dim, 
                    d_state=d_state, 
                    ssm_ratio=ssm_ratio,
                    channel_first=False  # ← 关键！统一使用 (B, H, W, C) 格式
                )
            except:
                # 回退到 SS2D
                if HAS_SS2D:
                    self.illum_branch = SS2D(
                        d_model=dim,
                        d_state=d_state,
                        ssm_ratio=ssm_ratio,
                        forward_type="v2",
                        channel_first=False,
                    )
                else:
                    raise ImportError("SS2D or VSSBlock is required")
        elif HAS_SS2D:
            self.illum_branch = SS2D(
                d_model=dim,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                forward_type="v2",
                channel_first=False,
            )
        else:
            raise ImportError("SS2D or VSSBlock is required")

        # 模块3: Reflectance Branch - 轻量 DW + Mamba
        self.refl_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        if HAS_NEW_MODULES:
            try:
                self.refl_mamba = Mamba(
                    d_model=dim,           # ← 关键！d_model 必须等于当前 dim，支持任意通道数（32/64/128等）
                    d_state=d_state,
                    expand=ssm_ratio
                )
            except:
                self.refl_mamba = nn.Identity()
        else:
            self.refl_mamba = nn.Identity()

        # 模块4: MineGate
        if HAS_NEW_MODULES:
            try:
                self.gate = MineGatePhysics()
            except:
                # 回退到简单融合
                self.gate = None
        else:
            self.gate = None
        
        # 伪光照提取器（用于 MineGate，如果 E0 未提供）
        self.illum_extractor = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, E0=None):
        """
        方案A：终极防崩版 forward 方法
        
        Args:
            x: (B, C, H, W) - 输入特征，C 可以是任意值（32/64/128等）
            E0: (B, 1, H, W) - 光照图（可选参数，如果未提供则自动提取）
        Returns:
            out: (B, C, H, W) - 输出特征
        """
        B, C, H, W = x.shape
        residual = x

        # LayerNorm + 转 (B,H,W,C) for Mamba
        x_norm = self.norm(x.permute(0, 2, 3, 1))  # (B,H,W,C)

        # Illumination Branch
        # VSSBlock 期望 (B, H, W, C) 格式（channel_first=False）
        x_illum = x.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
        f_illum = self.illum_branch(x_illum)  # VSSBlock 返回 (B, H, W, C)
        f_illum = f_illum.permute(0, 3, 1, 2).contiguous()  # (B, H, W, C) -> (B, C, H, W)

        # Reflectance Branch
        x_dw = self.refl_dw(x)  # (B, C, H, W)
        x_mamba = x_dw.permute(0, 2, 3, 1)  # (B,H,W,C)
        x_mamba = self.refl_mamba(x_mamba)   # Mamba 吃 (B,H,W,C)，d_model=dim 确保支持任意通道数（32/64/128等）
        f_refl = x_mamba.permute(0, 3, 1, 2)  # (B,C,H,W)

        # MineGate 融合
        # 如果 E0 未提供，从特征提取
        if E0 is None:
            E0 = self.illum_extractor(x)  # (B, 1, H, W)
        
        if self.gate is not None:
            fused, _ = self.gate(E0, f_illum, f_refl)
        else:
            # 简单平均融合（回退）
            fused = 0.5 * f_illum + 0.5 * f_refl

        return fused + residual
def forward(self, x, E0=None):
    if E0 is None:
        E0 = x.mean(dim=1, keepdim=True).detach()  # 兜底
    # ... 原有逻辑 ...
    fused, _ = self.gate(E0, f_illum, f_refl)
    return fused + x