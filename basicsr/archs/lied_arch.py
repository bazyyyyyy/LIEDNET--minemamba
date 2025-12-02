import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .vmamba import VSSLocalBlock

# 核心创新：使用 MineBlock 替换 VSSLocalBlock
try:
    from .mine_block import MineBlock
    USE_MINEBLOCK = True
except:
    USE_MINEBLOCK = False
    print("Warning: MineBlock not found, using VSSLocalBlock")

# Retinex 分解（如果存在）
try:
    from .retinex_decomp import StableRetinex
    USE_STABLE_RETINEX = True
except:
    USE_STABLE_RETINEX = False
    print("Warning: StableRetinex not found")

##########################################################################
## Overlapped image patch embedding with 3x3 Conv (Restormer style, no downsampling)
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- Restormer -----------------------
@ARCH_REGISTRY.register()
class LIED(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3,
        dim=64,                    # 改成64！性价比之王
        num_blocks=[2, 4, 6],       # 加深！免费提升1dB+
        num_refinement_blocks=6,   # 改成6！
        use_mineblock=True,
    ):
        super().__init__()

        # 1. Retinex 只负责提供 E0，不参与主干输入
        self.retinex = StableRetinex()  # 你原来的 StableRetinex

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        BlockClass = MineBlock
        print("Using MineBlock | dim=64 | blocks=[2,4,6] | refinement=6")

        # ====== 自动计算每层通道（关键！） ======
        c1 = dim           # 64
        c2 = dim * 4       # 256
        c3 = c2 * 4        # 1024

        # 自动生成降维卷积（防炸神器）
        # down1_2 实际输出：dim*2=128 通道（Conv2d减半到32，PixelUnshuffle乘以4到128）
        self.to_c2 = nn.Conv2d(dim*2, dim, 1)      # 128 → 64
        # down2_3 实际输出：dim*2=128 通道
        self.to_c3 = nn.Conv2d(dim*2, dim*2, 1)    # 128 → 128

        # ====== Encoder ======
        # 所有 MineBlock 都用合理通道
        self.encoder_level1 = nn.Sequential(*[BlockClass(dim=c1) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)                    # 64 → 256  正确
        self.encoder_level2 = nn.Sequential(*[BlockClass(dim=c1) for _ in range(num_blocks[1])])  # 输入 256 → 降到 64
        # down1_2 实际输出：64*2=128 通道（Conv2d减半到32，PixelUnshuffle乘以4到128）
        # 但经过 to_c2 压缩到 64，所以 down2_3 的输入是 64
        self.down2_3 = Downsample(dim)                    # 64 → 128  ← 实际是这个！
        self.encoder_level3 = nn.Sequential(*[BlockClass(dim=c1*2) for _ in range(num_blocks[2])]) # 输入 1024 → 降到 128
        self.bottleneck = BlockClass(dim=c1*2)

        # ====== Decoder ======
        # up3_2: 输入 dim*2 (128) → Conv2d(128, 256) → PixelShuffle(2) → 输出 64 通道（256/4=64）
        self.up3_2 = Upsample(c1*2)  # 128 → 64
        # Decoder skip 对齐: up3_2 输出 64 通道，已经是 c1，不需要 reduce
        # 但需要处理 skip connection 的通道对齐
        self.reduce2 = nn.Identity()  # 64 → 64（不需要降维）
        self.decoder_level2 = nn.Sequential(*[BlockClass(dim=c1) for _ in range(num_blocks[1])])  # 64

        # up2_1: 输入 dim (64) → Conv2d(64, 128) → PixelShuffle(2) → 输出 32 通道（128/4=32）
        self.up2_1 = Upsample(c1)  # 64 → 32
        # 但我们需要输出 64 通道，所以需要升维
        self.reduce1 = nn.Conv2d(c1//2, c1, 1)       # 32→64
        self.decoder_level1 = nn.Sequential(*[BlockClass(dim=c1) for _ in range(num_blocks[0])])  # 64

        # Refinement: 保持 c1 通道
        self.refinement = nn.Sequential(*[BlockClass(dim=c1) for _ in range(num_refinement_blocks)])  # 64

        # 输出层零初始化（你已经做了，保留）
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, inp_img, side_loss=False):
        # 关键！提取 E0，但主干吃原始输入
        E0, _ = self.retinex(inp_img)   # 只取 E0
        x = inp_img                      # 主干吃原始图

        # Encoder
        h1 = self.patch_embed(x)
        for blk in self.encoder_level1: h1 = blk(h1, E0)      # 传 E0！
        
        h2 = self.down1_2(h1)      # 64 → 256
        h2 = self.to_c2(h2)         # 256 → 64   ← 自动适配！
        for blk in self.encoder_level2: h2 = blk(h2, E0)
        
        h3 = self.down2_3(h2)      # 64 → 1024
        h3 = self.to_c3(h3)         # 1024 → 128 ← 自动适配！
        for blk in self.encoder_level3: h3 = blk(h3, E0)
        h = self.bottleneck(h3, E0)

        # Decoder
        x = self.up3_2(h)                        # 128 → 64
        x = self.reduce2(x)                      # 64 → 64（Identity，不需要降维）
        x = x + self.to_c2(self.down1_2(h1))      # skip connection: 128 → 64
        for blk in self.decoder_level2: x = blk(x, E0)

        x = self.up2_1(x)                        # 64 → 32
        x = self.reduce1(x)                      # 32 → 64 (升维对齐)
        x = x + h1                               # skip connection: 64 → 64
        for blk in self.decoder_level1: x = blk(x, E0)

        for blk in self.refinement: x = blk(x, E0)   # refinement也传E0！

        out = self.output(x) + inp_img
        return torch.clamp(out, 0, 1)