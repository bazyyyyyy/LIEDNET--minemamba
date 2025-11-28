"""
快速分析模型参数量和FLOPs分布
"""
import torch
from ptflops import get_model_complexity_info
from basicsr.archs.lied_arch import LIED
from basicsr.archs.vmamba import VSSLocalBlock, SS2D, LocalFeatureModule, DGDFeedForward

def analyze_model():
    model = LIED()
    model.eval()
    
    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")
    
    # 分析各个模块
    print("\n=== 模块参数量分析 ===")
    
    # VSSLocalBlock 单个分析
    block_32 = VSSLocalBlock(hidden_dim=32)
    block_64 = VSSLocalBlock(hidden_dim=64)
    block_128 = VSSLocalBlock(hidden_dim=128)
    
    params_32 = sum(p.numel() for p in block_32.parameters())
    params_64 = sum(p.numel() for p in block_64.parameters())
    params_128 = sum(p.numel() for p in block_128.parameters())
    
    print(f"VSSLocalBlock (32ch): {params_32 / 1e6:.3f}M")
    print(f"VSSLocalBlock (64ch): {params_64 / 1e6:.3f}M")
    print(f"VSSLocalBlock (128ch): {params_128 / 1e6:.3f}M")
    
    # SS2D 分析
    ssm_32 = SS2D(d_model=32, ssm_ratio=2.0, d_state=16)
    ssm_64 = SS2D(d_model=64, ssm_ratio=2.0, d_state=16)
    ssm_128 = SS2D(d_model=128, ssm_ratio=2.0, d_state=16)
    
    params_ssm_32 = sum(p.numel() for p in ssm_32.parameters())
    params_ssm_64 = sum(p.numel() for p in ssm_64.parameters())
    params_ssm_128 = sum(p.numel() for p in ssm_128.parameters())
    
    print(f"\nSS2D (32ch): {params_ssm_32 / 1e6:.3f}M")
    print(f"SS2D (64ch): {params_ssm_64 / 1e6:.3f}M")
    print(f"SS2D (128ch): {params_ssm_128 / 1e6:.3f}M")
    
    # DGDFFN 分析
    ffn_32 = DGDFeedForward(dim=32, ffn_expansion_factor=2.66, bias=False)
    ffn_64 = DGDFeedForward(dim=64, ffn_expansion_factor=2.66, bias=False)
    ffn_128 = DGDFeedForward(dim=128, ffn_expansion_factor=2.66, bias=False)
    
    params_ffn_32 = sum(p.numel() for p in ffn_32.parameters())
    params_ffn_64 = sum(p.numel() for p in ffn_64.parameters())
    params_ffn_128 = sum(p.numel() for p in ffn_128.parameters())
    
    print(f"\nDGDFFN (32ch): {params_ffn_32 / 1e6:.3f}M")
    print(f"DGDFFN (64ch): {params_ffn_64 / 1e6:.3f}M")
    print(f"DGDFFN (128ch): {params_ffn_128 / 1e6:.3f}M")
    
    # LFM 分析
    lfm_32 = LocalFeatureModule(dim=32)
    lfm_64 = LocalFeatureModule(dim=64)
    lfm_128 = LocalFeatureModule(dim=128)
    
    params_lfm_32 = sum(p.numel() for p in lfm_32.parameters())
    params_lfm_64 = sum(p.numel() for p in lfm_64.parameters())
    params_lfm_128 = sum(p.numel() for p in lfm_128.parameters())
    
    print(f"\nLFM (32ch): {params_lfm_32 / 1e6:.3f}M")
    print(f"LFM (64ch): {params_lfm_64 / 1e6:.3f}M")
    print(f"LFM (128ch): {params_lfm_128 / 1e6:.3f}M")
    
    # 估算各层级参数量
    print("\n=== 层级参数量估算 ===")
    print(f"Encoder L1 (1 block × 32ch): {params_32 / 1e6:.3f}M")
    print(f"Encoder L2 (2 blocks × 64ch): {params_64 * 2 / 1e6:.3f}M")
    print(f"Encoder L3 (4 blocks × 128ch): {params_128 * 4 / 1e6:.3f}M")
    print(f"Bottleneck (1 block × 128ch): {params_128 / 1e6:.3f}M")
    print(f"Decoder L2 (2 blocks × 64ch): {params_64 * 2 / 1e6:.3f}M")
    print(f"Decoder L1 (1 block × 64ch): {params_64 / 1e6:.3f}M")
    print(f"Refinement (4 blocks × 64ch): {params_64 * 4 / 1e6:.3f}M")
    
    # FLOPs 分析
    print("\n=== FLOPs 分析 ===")
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"模型总 FLOPs: {macs}")
    print(f"模型总参数量: {params}")

if __name__ == "__main__":
    analyze_model()
