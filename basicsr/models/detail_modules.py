#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量细节增强模块：多尺度卷积 + 梯度引导 + 动态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientExtractor(nn.Module):
    """Sobel 梯度提取器，缓存卷积核避免重复创建。"""
    _kernel_cache = {}

    def __init__(self, in_channels=3):
        super().__init__()
        if in_channels not in GradientExtractor._kernel_cache:
            kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            kernel_y = kernel_x.T
            weight_x = kernel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            weight_y = kernel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            GradientExtractor._kernel_cache[in_channels] = (weight_x, weight_y)
        weight_x, weight_y = GradientExtractor._kernel_cache[in_channels]
        self.register_buffer('weight_x', weight_x, persistent=False)
        self.register_buffer('weight_y', weight_y, persistent=False)

    def forward(self, x):
        weight_x = self.weight_x.to(x.device, x.dtype)
        weight_y = self.weight_y.to(x.device, x.dtype)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, weight_y, padding=1, groups=x.size(1))
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad


class MultiScaleConv(nn.Module):
    """降维 + 深度可分离卷积实现的多尺度卷积。"""

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(1, in_channels // 2)
        self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
        self.dw5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels, bias=False)
        self.act = nn.GELU()
        self.pw3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.pw5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.fuse = nn.Conv2d(mid_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        reduced = self.reduce(x)
        b3 = self.pw3(self.act(self.dw3(reduced)))
        b5 = self.pw5(self.act(self.dw5(reduced)))
        fused = torch.cat([b3, b5], dim=1)
        return self.fuse(fused)


class ResidualGate(nn.Module):
    """轻量瓶颈门控，用于抑制增强残差。"""

    def __init__(self, in_channels):
        super().__init__()
        hidden = max(1, in_channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, residual):
        return self.conv(residual)


class DynamicFusion(nn.Module):
    """自适应融合原图、增强图与边缘图。"""

    def __init__(self, in_channels):
        super().__init__()
        hidden = max(3, in_channels // 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_gen = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x, enhanced, edge):
        B, C, H, W = x.shape
        fusion_avg = self.avg_pool(torch.cat([x, enhanced, edge], dim=1))
        weights = self.weight_gen(fusion_avg).view(B, 3, 1, 1, 1)
        fusion_input = torch.stack([x, enhanced, edge], dim=1)
        fused = (fusion_input * weights).sum(dim=1)
        return fused


class MultiScaleDetailEnhancer(nn.Module):
    """轻量细节增强模块：多尺度卷积 + 梯度引导 + 动态融合。"""

    def __init__(self, in_channels=3):
        super().__init__()
        mid_channels = max(1, in_channels // 2)
        self.ms_conv = MultiScaleConv(in_channels)
        self.edge_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.gradient_extractor = GradientExtractor(in_channels)
        self.residual_gate = ResidualGate(in_channels)
        self.dynamic_fusion = DynamicFusion(in_channels)

    def forward(self, x):
        enhanced_features = self.ms_conv(x)
        gate = self.residual_gate(enhanced_features - x)
        enhanced_features = enhanced_features * gate

        grad_map = self.gradient_extractor(x)
        edge_features = self.edge_conv(self.edge_reduce(grad_map))

        out = self.dynamic_fusion(x, enhanced_features, edge_features)
        return out 