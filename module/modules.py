"""
模块3的核心组件：MambaBlock 和 ChannelMambaBlock
用于替换 VSSLocalBlock 中的 SS2D 和 LocalFeatureModule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from basicsr.archs.vmamba import SS2D, CrossScan, CrossMerge
    from basicsr.archs.vmamba import selective_scan_cuda_core
    HAS_SELECTIVE_SCAN = True
except:
    HAS_SELECTIVE_SCAN = False
    print("Warning: selective_scan_cuda_core not found, using fallback")


class DepthwiseConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MambaBlock(nn.Module):
    """
    轻量化的 Mamba Block（替换 SS2D）
    
    设计思路：
    - 比 SS2D 更轻量（可能只用单方向或双方向扫描，而不是4方向）
    - 保持 Mamba 的核心 SelectiveScan 机制
    - 减少参数量和计算量
    
    输入输出：(B, C, H, W) 或 (B, H, W, C)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        ssm_ratio: float = 1.5,  # 比 SS2D 的 2.0 更小，减少参数
        dt_rank: Optional[int] = None,
        d_conv: int = 3,
        act_layer=nn.SiLU,
        channel_first: bool = True,  # 默认 channel_first
        **kwargs
    ):
        super().__init__()
        self.channel_first = channel_first
        dt_rank = dt_rank or max(16, d_model // 16)
        d_inner = int(d_model * ssm_ratio)
        
        # 使用简化的 SS2D（单方向或双方向，而不是4方向）
        # 这里我们使用 SS2D 但配置更轻量
        self.ssm = SS2D(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,  # 1.5 而不是 2.0
            dt_rank=dt_rank,
            d_conv=d_conv,
            act_layer=act_layer,
            forward_type="v2",
            channel_first=channel_first,
            **kwargs
        )
        
        # 可选的轻量归一化
        if channel_first:
            self.norm = nn.BatchNorm2d(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) if channel_first else (B, H, W, C)
        Returns:
            out: 同尺寸
        """
        if self.channel_first:
            # (B, C, H, W) -> (B, H, W, C) for SS2D
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.ssm(x)  # (B, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            x = self.norm(x)
        else:
            x = self.ssm(x)  # (B, H, W, C)
            x = self.norm(x)
        
        return x


class ChannelMambaBlock(nn.Module):
    """
    通道维度的 Mamba Block（替换 LocalFeatureModule）
    
    设计思路：
    - 将通道维度作为序列，使用 Mamba 建模通道间依赖
    - 比 LocalFeatureModule 的注意力机制更轻量
    - 保持通道建模能力
    
    输入输出：(B, C, H, W)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 8,  # 通道维度通常比空间维度小，用更小的 d_state
        ssm_ratio: float = 1.0,  # 通道维度不需要太大扩展
        dt_rank: Optional[int] = None,
        act_layer=nn.SiLU,
        **kwargs
    ):
        super().__init__()
        dt_rank = dt_rank or max(8, d_model // 16)
        d_inner = int(d_model * ssm_ratio)
        
        # 通道维度的投影
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.act = act_layer()
        
        # 通道维度的 SSM（将每个空间位置独立处理，通道作为序列）
        # 这里我们使用轻量化的 SS2D，但需要特殊处理通道维度
        self.channel_ssm = SS2D(
            d_model=d_inner,
            d_state=d_state,
            ssm_ratio=1.0,  # 通道维度不需要扩展
            dt_rank=dt_rank,
            d_conv=1,  # 通道维度不需要卷积
            act_layer=act_layer,
            forward_type="v2",
            channel_first=False,  # 通道维度用 channel_last
            **kwargs
        )
        
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 归一化
        x_norm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm = self.norm(x_norm)  # (B, H, W, C)
        
        # 投影
        x_proj = self.in_proj(x_norm)  # (B, H, W, 2*d_inner)
        x1, x2 = x_proj.chunk(2, dim=-1)  # (B, H, W, d_inner) each
        
        # 通道维度的 SSM
        # 对每个空间位置 (h, w)，将通道维度 C 作为序列长度
        # 需要 reshape: (B, H, W, d_inner) -> (B*H*W, 1, 1, d_inner) -> SSM
        x1_reshaped = x1.view(B * H * W, 1, 1, -1)  # (B*H*W, 1, 1, d_inner)
        x1_ssm = self.channel_ssm(x1_reshaped)  # (B*H*W, 1, 1, d_inner)
        x1_ssm = x1_ssm.view(B, H, W, -1)  # (B, H, W, d_inner)
        
        # Gating
        x2 = self.act(x2)  # (B, H, W, d_inner)
        
        # 融合
        x_out = x1_ssm * x2  # (B, H, W, d_inner)
        x_out = self.out_proj(x_out)  # (B, H, W, C)
        
        # 残差连接
        x_out = x_out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x + x_out


# 为了兼容性，也可以提供一个更轻量的版本
class LightweightMambaBlock(nn.Module):
    """
    超轻量化的 Mamba Block（如果 MambaBlock 还是太重）
    只使用单方向扫描
    """
    def __init__(self, d_model: int, d_state: int = 16, channel_first: bool = True):
        super().__init__()
        self.channel_first = channel_first
        d_inner = int(d_model * 1.2)  # 更小的扩展
        
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.act = nn.SiLU()
        
        # 简化的 SSM（单方向）
        self.ssm = SS2D(
            d_model=d_inner,
            d_state=d_state,
            ssm_ratio=1.0,
            forward_type="v31d",  # 单方向扫描
            channel_first=channel_first,
        )
        
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model) if not channel_first else nn.BatchNorm2d(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()
        x = self.in_proj(x)
        x = self.act(x)
        x = self.ssm(x)
        x = self.out_proj(x)
        if self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x




