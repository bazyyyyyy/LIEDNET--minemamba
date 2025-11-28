"""
VSSBlock - Visual State Space Block
用于替换 VSSLocalBlock 中的 SS2D (VSSM)
实现全局 State Space Model，支持4方向扫描
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from basicsr.archs.vmamba import SS2D
    from timm.models.layers import DropPath
    HAS_SS2D = True
except:
    HAS_SS2D = False
    print("Warning: SS2D not found")


class PatchEmbed(nn.Module):
    """Patch Embedding for VSSBlock"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        return x


class VSSBlock(nn.Module):
    """
    VSSBlock - Visual State Space Block
    
    对应架构图中的 VSSM (Visual State Space Model)
    实现全局 State Space Model，支持4方向扫描
    
    结构：
    - Linear → Dw3x3 → SiLU → 2D-SSM → LN
    - Linear → SiLU
    - Matrix Multiplication
    - Linear
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        dt_rank: Optional[int] = None,
        d_conv: int = 3,
        act_layer=nn.SiLU,
        drop_path: float = 0.0,
        scan_direction: str = 'all',  # 'all' for 4-direction, 'h' for horizontal, 'v' for vertical
        channel_first: bool = False,  # VSSBlock 默认使用 channel_last
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.scan_direction = scan_direction
        self.channel_first = channel_first
        
        dt_rank = dt_rank or max(16, dim // 16)
        d_inner = int(dim * ssm_ratio)
        
        # 使用 SS2D 作为核心 State Space Model
        if HAS_SS2D:
            # 根据 scan_direction 选择 forward_type
            if scan_direction == 'all':
                forward_type = "v2"  # 4方向扫描
            elif scan_direction == 'h':
                forward_type = "v32d"  # 2方向（水平+垂直）
            elif scan_direction == 'v':
                forward_type = "v31d"  # 单方向
            else:
                forward_type = "v2"
            
            self.ssm = SS2D(
                d_model=dim,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                d_conv=d_conv,
                act_layer=act_layer,
                forward_type=forward_type,
                channel_first=channel_first,
                **kwargs
            )
        else:
            raise ImportError("SS2D is required for VSSBlock")
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
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
            x = self.drop_path(self.ssm(x))  # (B, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        else:
            # (B, H, W, C)
            x = self.drop_path(self.ssm(x))  # (B, H, W, C)
        
        return x


# 为了兼容性，提供一个轻量版本
class LightweightVSSBlock(nn.Module):
    """
    轻量化的 VSSBlock（如果标准版本太重）
    使用更小的 ssm_ratio 和单方向扫描
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        ssm_ratio: float = 1.5,  # 更小的扩展比
        drop_path: float = 0.0,
        channel_first: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vss_block = VSSBlock(
            dim=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            drop_path=drop_path,
            scan_direction='h',  # 单方向扫描，更轻量
            channel_first=channel_first,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vss_block(x)