"""
使用示例：如何在 LIED 架构中启用新模块
"""

from basicsr.archs.lied_arch import LIED
from basicsr.archs.vmamba import VSSLocalBlock

# ============================================
# 示例1：在 LIED 架构中启用新模块
# ============================================

# 修改 lied_arch.py 中的 VSSLocalBlock 创建
# 在 LIED.__init__ 中：

def create_lied_with_new_modules():
    """创建使用新模块的 LIED 模型"""
    
    # 方法1：直接修改 LIED 类
    class LIED_Enhanced(LIED):
        def __init__(self, 
            inp_channels=3, 
            out_channels=3,
            dim=32,
            num_blocks=[1, 2, 4], 
            num_refinement_blocks=4,
            use_vssblock=False,      # 启用 VSSBlock
            use_channel_mamba=False, # 启用 ChannelMambaBlock
            use_minegate=True,      # 启用 MineGate（默认已启用）
        ):
            super().__init__(inp_channels, out_channels, dim, num_blocks, num_refinement_blocks)
            
            # 重新创建 encoder_level1，使用新模块
            self.encoder_level1 = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=dim,
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_blocks[0])
            ])
            
            # 同样修改其他 level
            self.encoder_level2 = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=int(dim*2**1),
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_blocks[1])
            ])
            
            self.encoder_level3 = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=int(dim*2**2),
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_blocks[2])
            ])
            
            # Bottleneck
            self.bottleneck = VSSLocalBlock(
                hidden_dim=int(dim*2**2),
                decoder=False,
                use_vssblock=use_vssblock,
                use_channel_mamba=use_channel_mamba,
                use_minegate=use_minegate,
            )
            
            # Decoder levels
            self.decoder_level2 = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=int(dim*2**1),
                    decoder=True,
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_blocks[1])
            ])
            
            self.decoder_level1 = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=int(dim*2**1),
                    decoder=True,
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_blocks[0])
            ])
            
            self.refinement = nn.Sequential(*[
                VSSLocalBlock(
                    hidden_dim=int(dim*2**1),
                    decoder=True,
                    use_vssblock=use_vssblock,
                    use_channel_mamba=use_channel_mamba,
                    use_minegate=use_minegate,
                ) 
                for i in range(num_refinement_blocks)
            ])
    
    return LIED_Enhanced


# ============================================
# 示例2：单独测试 VSSLocalBlock
# ============================================

def test_vsslocalblock():
    """测试不同配置的 VSSLocalBlock"""
    import torch
    
    # 原始版本（默认）
    block_original = VSSLocalBlock(hidden_dim=32)
    
    # 只启用 ChannelMambaBlock（推荐先测试这个）
    block_lfm = VSSLocalBlock(hidden_dim=32, use_channel_mamba=True)
    
    # 只启用 VSSBlock
    block_vssm = VSSLocalBlock(hidden_dim=32, use_vssblock=True)
    
    # 全部启用
    block_all = VSSLocalBlock(
        hidden_dim=32,
        use_vssblock=True,
        use_channel_mamba=True,
        use_minegate=True,
    )
    
    # 测试前向传播
    x = torch.randn(2, 32, 64, 64)  # (B, C, H, W)
    x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C) for VSSLocalBlock
    
    print("Testing VSSLocalBlock...")
    try:
        out_original = block_original(x)
        print("✓ Original block works")
        
        out_lfm = block_lfm(x)
        print("✓ Block with ChannelMambaBlock works")
        
        out_vssm = block_vssm(x)
        print("✓ Block with VSSBlock works")
        
        out_all = block_all(x)
        print("✓ Block with all new modules works")
        
        print(f"Output shapes: {out_original.shape}, {out_lfm.shape}, {out_vssm.shape}, {out_all.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")


# ============================================
# 示例3：在训练脚本中使用
# ============================================

def modify_training_script():
    """
    在 train_llie.py 中修改模型创建部分
    
    原来的代码：
    model_restored = LIED()
    
    修改为：
    """
    # 方式1：直接创建增强版
    # from example_usage import create_lied_with_new_modules
    # LIED_Enhanced = create_lied_with_new_modules()
    # model_restored = LIED_Enhanced(
    #     use_vssblock=True,
    #     use_channel_mamba=True,
    #     use_minegate=True,
    # )
    
    # 方式2：修改原始 LIED 类（推荐）
    # 直接修改 basicsr/archs/lied_arch.py 中的 VSSLocalBlock 创建
    # 添加参数传递
    pass


if __name__ == "__main__":
    # 运行测试
    test_vsslocalblock()




