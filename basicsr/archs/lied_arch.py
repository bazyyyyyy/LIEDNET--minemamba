import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from .vmamba import VSSLocalBlock

# 核心创新：使用 MineBlock 替换 VSSLocalBlock
try:
    from .mine_block import MineBlock
    USE_MINEBLOCK = True
except:
    USE_MINEBLOCK = False
    print("Warning: MineBlock not found, using VSSLocalBlock")

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
        dim = 32,
        num_blocks = [1, 2, 4], 
        num_refinement_blocks = 4,
        use_mineblock: bool = True,  # 核心创新：使用 MineBlock 替换 VSSLocalBlock
    ):

        super(LIED, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 核心创新：使用 MineBlock 替换所有 VSSLocalBlock
        if use_mineblock and USE_MINEBLOCK:
            BlockClass = MineBlock
            print("==> Using MineBlock (Illum VMamba + Refl DW+Mamba + MineGate)")
            print(f"==> Channel alignment: Level1={dim}, Level2={int(dim*2**1)}, Level3={int(dim*2**2)}")
            
            # MineBlock 使用 dim 参数，自动适配通道数
            self.encoder_level1 = nn.Sequential(*[BlockClass(dim=dim) for i in range(num_blocks[0])])
            
            self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
            self.encoder_level2 = nn.Sequential(*[BlockClass(dim=int(dim*2**1)) for i in range(num_blocks[1])])
            
            self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
            self.encoder_level3 = nn.Sequential(*[BlockClass(dim=int(dim*2**2)) for i in range(num_blocks[2])])

            # Bottleneck
            self.bottleneck = BlockClass(dim=int(dim*2**2))

            # Decoder: 3 levels (Restormer style with skip connections)
            self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
            self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=False)  # Cat后4C -> 2C
            self.decoder_level2 = nn.Sequential(*[BlockClass(dim=int(dim*2**1)) for i in range(num_blocks[1])])
            
            self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1
            self.decoder_level1 = nn.Sequential(*[BlockClass(dim=int(dim*2**1)) for i in range(num_blocks[0])])
            
            self.refinement = nn.Sequential(*[BlockClass(dim=int(dim*2**1)) for i in range(num_refinement_blocks)])
        else:
            BlockClass = VSSLocalBlock
            print("==> Using VSSLocalBlock (original)")
            
            self.encoder_level1 = nn.Sequential(*[BlockClass(hidden_dim=dim, channel_first=False) for i in range(num_blocks[0])])
            
            self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
            self.encoder_level2 = nn.Sequential(*[BlockClass(hidden_dim=int(dim*2**1), channel_first=False) for i in range(num_blocks[1])])
            
            self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
            self.encoder_level3 = nn.Sequential(*[BlockClass(hidden_dim=int(dim*2**2), channel_first=False) for i in range(num_blocks[2])])

            # Bottleneck / Feature Fusion (F₃ → MDTA → GDFN → F₃')
            self.bottleneck = BlockClass(hidden_dim=int(dim*2**2), channel_first=False)

            # Decoder: 3 levels (Restormer style with skip connections)
            self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
            self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=False)  # Cat后4C -> 2C
            self.decoder_level2 = nn.Sequential(*[BlockClass(hidden_dim=int(dim*2**1), channel_first=False) for i in range(num_blocks[1])])
            
            self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1
            self.decoder_level1 = nn.Sequential(*[BlockClass(hidden_dim=int(dim*2**1), channel_first=False) for i in range(num_blocks[0])])
            
            self.refinement = nn.Sequential(*[BlockClass(hidden_dim=int(dim*2**1), channel_first=False) for i in range(num_refinement_blocks)])
        
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.side_out = nn.Conv2d(128, 3, 3, stride=1, padding=1)

    def forward(self, inp_img, side_loss=False):
        # 检测是否使用 MineBlock（通过检查第一个 block 的类型）
        use_mineblock = USE_MINEBLOCK and hasattr(self, 'encoder_level1') and len(self.encoder_level1) > 0
        if use_mineblock:
            try:
                use_mineblock = isinstance(self.encoder_level1[0], MineBlock)
            except:
                use_mineblock = False
        
        if use_mineblock:
            # ========== MineBlock 路径（方案A：终极防崩版）==========
            # MineBlock 直接接受 (B, C, H, W) 格式，自动处理格式转换
            # E0 参数：如果未提供，MineBlock 内部会自动从特征提取
            
            # Encoder Level 1
            inp_enc_level1 = self.patch_embed(inp_img)  # (B, C, H, W)
            out_enc_level1 = inp_enc_level1
            for block in self.encoder_level1:
                out_enc_level1 = block(out_enc_level1, None)  # (B, C, H, W), E0=None 自动提取
            
            # Encoder Level 2
            inp_enc_level2 = self.down1_2(out_enc_level1)  # (B, 2C, H/2, W/2)
            out_enc_level2 = inp_enc_level2
            for block in self.encoder_level2:
                out_enc_level2 = block(out_enc_level2, None)  # (B, 2C, H/2, W/2), E0=None 自动提取
            
            # Encoder Level 3
            inp_enc_level3 = self.down2_3(out_enc_level2)  # (B, 4C, H/4, W/4)
            out_enc_level3 = inp_enc_level3
            for block in self.encoder_level3:
                out_enc_level3 = block(out_enc_level3, None)  # (B, 4C, H/4, W/4), E0=None 自动提取
            
            if side_loss:
                out_side = self.side_out(out_enc_level3)
            
            # Bottleneck
            bottleneck_output = self.bottleneck(out_enc_level3, None)  # (B, 4C, H/4, W/4), E0=None 自动提取
            
            # Decoder Level 2
            inp_dec_level2 = self.up3_2(bottleneck_output)  # (B, 2C, H/2, W/2)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # (B, 4C, H/2, W/2) - Skip connection
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # (B, 2C, H/2, W/2)
            out_dec_level2 = inp_dec_level2
            for block in self.decoder_level2:
                out_dec_level2 = block(out_dec_level2, None)  # (B, 2C, H/2, W/2), E0=None 自动提取
            
            # Decoder Level 1
            inp_dec_level1 = self.up2_1(out_dec_level2)  # (B, C, H, W)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # (B, 2C, H, W) - Skip connection
            out_dec_level1 = inp_dec_level1
            for block in self.decoder_level1:
                out_dec_level1 = block(out_dec_level1, None)  # (B, 2C, H, W), E0=None 自动提取
            
            # Refinement
            for block in self.refinement:
                out_dec_level1 = block(out_dec_level1, None)  # (B, 2C, H, W), E0=None 自动提取
        else:
            # ========== VSSLocalBlock 路径（原始格式）==========
            # Encoder path (Restormer style: 3 levels, no level 4)
            inp_enc_level1 = self.patch_embed(inp_img).permute(0, 2, 3, 1).contiguous()  # (B, H, W, C) - 不降采样
            out_enc_level1 = self.encoder_level1(inp_enc_level1).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            
            inp_enc_level2 = self.down1_2(out_enc_level1).permute(0, 2, 3, 1).contiguous()  # (B, H/2, W/2, 2C)
            out_enc_level2 = self.encoder_level2(inp_enc_level2).permute(0, 3, 1, 2).contiguous()  # (B, 2C, H/2, W/2)

            inp_enc_level3 = self.down2_3(out_enc_level2).permute(0, 2, 3, 1).contiguous()  # (B, H/4, W/4, 4C)
            out_enc_level3 = self.encoder_level3(inp_enc_level3).permute(0, 3, 1, 2).contiguous()  # (B, 4C, H/4, W/4)
            
            if side_loss:
                out_side = self.side_out(out_enc_level3)

            # Bottleneck / Feature Fusion: F₃ → MDTA → GDFN → F₃'
            bottleneck_input = out_enc_level3.permute(0, 2, 3, 1).contiguous()  # (B, H/4, W/4, 4C)
            bottleneck_output = self.bottleneck(bottleneck_input).permute(0, 3, 1, 2).contiguous()  # (B, 4C, H/4, W/4) → F₃'

            # Decoder path (Restormer style: 3 levels with skip connections)
            inp_dec_level2 = self.up3_2(bottleneck_output)  # (B, 2C, H/2, W/2) - Upsample输出是输入的一半
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # (B, 4C, H/2, W/2) - Skip connection
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2).permute(0, 2, 3, 1).contiguous()  # (B, 2C, H/2, W/2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2).permute(0, 3, 1, 2).contiguous()  # (B, 2C, H/2, W/2)

            inp_dec_level1 = self.up2_1(out_dec_level2)  # (B, 1C, H, W) - Upsample输出是输入的一半
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # (B, 2C, H, W) - Skip connection (NO 1x1 conv)
            out_dec_level1 = self.decoder_level1(inp_dec_level1.permute(0, 2, 3, 1).contiguous())
            
            out_dec_level1 = self.refinement(out_dec_level1).permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)

        # Output (Restormer style: direct output without recover)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        if side_loss:
            return out_side, out_dec_level1
        else:
            return out_dec_level1