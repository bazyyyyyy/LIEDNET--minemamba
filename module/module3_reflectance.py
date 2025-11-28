"""
模块3: Reflectance Branch (局部反射率分支)
独立模块文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .modules import DepthwiseConv2d, MambaBlock, ChannelMambaBlock


class ReflectanceBranch(nn.Module):
    """
    模块3: Reflectance Branch (局部反射率分支)
    输入: R0 (B, 3, H, W)
    - 每级先 Depthwise 3x3 Conv
    - 再接一个 Mamba 块
    - 再接 Channel-wise Mamba
    - 4级下采样 + 4级上采样
    输出: F_refl (B, 64, H, W)
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = num_levels

        self.channels = [base_channels * (2 ** i) for i in range(num_levels)]

        # 初始卷积
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder stages
        self.encoder = nn.ModuleList()
        for ch in self.channels:
            self.encoder.append(nn.ModuleList([
                DepthwiseConv2d(ch, ch, kernel_size=3, padding=1),
                nn.SiLU(),
                MambaBlock(d_model=ch),
                ChannelMambaBlock(d_model=ch),
            ]))

        # 下采样层（通道翻倍）
        self.downsamples = nn.ModuleList()
        for i in range(num_levels - 1):
            self.downsamples.append(
                nn.Sequential(
                    nn.Conv2d(self.channels[i], self.channels[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.SiLU(),
                )
            )

        # Decoder stages（逆序）
        self.decoder = nn.ModuleList()
        for i in range(num_levels - 1, 0, -1):
            in_ch = self.channels[i]
            skip_ch = self.channels[i - 1]
            self.decoder.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, skip_ch, kernel_size=3, padding=1),
                nn.SiLU(),
                MambaBlock(d_model=skip_ch),
                ChannelMambaBlock(d_model=skip_ch),
            ]))
    
    def forward(self, R0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R0: (B, 3, H, W) 粗反射率
        Returns:
            F_refl: (B, 64, H, W) 反射率特征
        """
        B, C, H, W = R0.shape
        
        # 初始卷积
        x = self.input_conv(R0)  # (B, 64, H, W)
        
        # Encoder: 4级下采样
        skip_connections = []
        for i, (encoder_stage, downsample) in enumerate(zip(self.encoder[:-1], self.downsamples)):
            # Encoder stage
            x = encoder_stage[0](x)  # DWConv
            x = encoder_stage[1](x)  # SiLU
            x = encoder_stage[2](x)  # MambaBlock
            x = encoder_stage[3](x)  # ChannelMambaBlock
            
            skip_connections.append(x)
            
            # Downsample
            x = downsample(x)
        
        # 最深层（无下采样）
        x = self.encoder[-1][0](x)
        x = self.encoder[-1][1](x)
        x = self.encoder[-1][2](x)
        x = self.encoder[-1][3](x)
        
        # Decoder: 4级上采样
        for i, decoder_stage in enumerate(self.decoder):
            # Upsample
            x = decoder_stage[0](x)
            x = decoder_stage[1](x)  # Conv
            x = decoder_stage[2](x)  # SiLU
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip
            
            # Mamba blocks
            x = decoder_stage[3](x)  # MambaBlock
            x = decoder_stage[4](x)  # ChannelMambaBlock
        
        # 确保输出尺寸匹配输入
        if x.shape[2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x

