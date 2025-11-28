"""
模块2: Illumination Branch (全局光照分支)
独立模块文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .vss_block import VSSBlock, PatchEmbed


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensors"""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class IlluminationBranch(nn.Module):
    """
    模块2: Illumination Branch (全局光照分支)
    输入: E0 (B, 1, H, W)
    - PatchEmbed (conv 4x4 stride 4)
    - 4级 VMamba（每级2个 VSSBlock，四方向扫描）
    - 上采样回原分辨率
    输出: F_illum (B, 64, H, W)
    """
    def __init__(
        self,
        embed_dim: int = 64,
        depths: list = [2, 2, 2, 2],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            patch_size=4,
            in_chans=1,
            embed_dim=embed_dim,
        )
        
        # 4级 VMamba
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i, depth in enumerate(depths):
            stage = nn.ModuleList()
            for j in range(depth):
                stage.append(
                    VSSBlock(
                        dim=embed_dim,
                        drop_path=dpr[cur + j],
                        scan_direction='all',
                    )
                )
            self.stages.append(stage)
            cur += depth
        
        # 下采样层（3个，用于4级中的前3级）
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
                LayerNorm2d(embed_dim),
            )
            for _ in range(len(depths) - 1)
        ])
        
        # 上采样回原分辨率
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            ) for _ in range(len(depths) - 1)
        ])
        
        # 最终输出层（上采样4倍回到原分辨率）
        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
    
    def forward(self, E0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E0: (B, 1, H, W) 粗光照
        Returns:
            F_illum: (B, 64, H, W) 光照特征
        """
        B, C, H, W = E0.shape
        
        # Patch Embedding
        x = self.patch_embed(E0)  # (B, embed_dim, H//4, W//4)
        skip_connections = []
        
        # 4级下采样 + VSSBlock
        for i, stage in enumerate(self.stages):
            # VSSBlock 阶段
            for block in stage:
                x = block(x)
            skip_connections.append(x)
            
            # 下采样（除了最后一级）
            if i < len(self.stages) - 1:
                x = self.downsample_layers[i](x)
        
        # 上采样 + 融合 skip connections
        for i, upsample in enumerate(self.upsample_layers):
            x = upsample(x)
            # 融合对应的 skip connection
            skip = skip_connections[-(i+2)]
            # 调整 skip 的尺寸
            if x.shape != skip.shape:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
        
        # 最终上采样到原分辨率
        x = self.output_conv(x)
        
        # 确保输出尺寸匹配输入
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

