"""
模块5: U-Net Decoder + VSS Skip Fusion
独立模块文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .vss_block import VSSBlock


class UNetDecoder(nn.Module):
    """
    模块5: U-Net Decoder + VSS Skip Fusion
    U-Net style 4级 upsample (ch=128→64→32→16) + skip from branches (via VSSBlock fusion)
    → final 1x1 Conv to 3ch residual
    All SiLU, LayerNorm pre/post
    """
    def __init__(
        self,
        in_channels: int = 64,
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # Channel progression: 128 → 64 → 32 → 16
        channels = [128, 64, 32, 16]
        
        # Decoder layers: 4级上采样
        self.decoder_layers = nn.ModuleList()
        
        # First level: 128 → 64
        self.decoder_layers.append(nn.ModuleDict({
            'norm_pre': nn.LayerNorm(in_channels),
            'upsample': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, channels[1], kernel_size=3, padding=1, bias=False),
            ),
            'norm_post': nn.LayerNorm(channels[1]),
            'vss_fusion': VSSBlock(dim=channels[1], scan_direction='all'),
            'skip_proj': nn.Conv2d(64, channels[1], kernel_size=1, bias=False),  # 64 is embed_dim
            'conv': nn.Sequential(
                nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1, bias=False),
            ),
            'act': nn.SiLU(),
        }))
        
        # Levels 2-4: 64 → 32 → 16
        for i in range(1, num_levels - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            self.decoder_layers.append(nn.ModuleDict({
                'norm_pre': nn.LayerNorm(in_ch),
                'upsample': nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                ),
                'norm_post': nn.LayerNorm(out_ch),
                'vss_fusion': VSSBlock(dim=out_ch, scan_direction='all'),
                'skip_proj': nn.Conv2d(64, out_ch, kernel_size=1, bias=False),  # 64 is embed_dim
                'conv': nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                ),
                'act': nn.SiLU(),
            }))
        
        # 最终输出层: 16 → 3 (1x1 conv)
        self.output_norm = nn.LayerNorm(channels[-1])
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels[-1], 3, kernel_size=1, bias=False),
        )
    
    def forward(self, F_fused: torch.Tensor, skip_connections: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            F_fused: (B, 64, H, W) 融合特征
            skip_connections: List of skip connection features from branches
        Returns:
            residual: (B, 3, H, W) 残差输出
        """
        x = F_fused
        B, C, H, W = x.shape
        
        # 上采样并融合 skip connections
        for i, layer in enumerate(self.decoder_layers):
            # Pre-norm
            x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x_norm = layer['norm_pre'](x_norm)
            x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Upsample
            x = layer['upsample'](x_norm)
            
            # Post-norm
            x_norm = x.permute(0, 2, 3, 1)
            x_norm = layer['norm_post'](x_norm)
            x_norm = x_norm.permute(0, 3, 1, 2)
            
            # 融合 skip connection（如果有）
            if i < len(skip_connections) and skip_connections[-(i+1)] is not None:
                skip = skip_connections[-(i+1)]
                if x_norm.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x_norm.shape[2:], mode='bilinear', align_corners=False)
                skip = layer['skip_proj'](skip)
                x_norm = x_norm + skip
            
            # VSSBlock 融合
            x = layer['vss_fusion'](x_norm)
            
            # Conv + Activation
            x = layer['conv'](x)
            x = layer['act'](x)
        
        # 最终输出 (1x1 conv to 3ch)
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.output_norm(x_norm)  # LayerNorm
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.output_conv[0](x_norm)  # Conv 3x3
        x = self.output_conv[1](x)  # SiLU
        residual = self.output_conv[2](x)  # Conv 1x1 to 3ch
        
        return residual

