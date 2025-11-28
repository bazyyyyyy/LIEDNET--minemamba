"""
测试 MineBlock 实现
"""
import torch
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from basicsr.archs.mine_block import MineBlock
    print("✓ 成功导入 MineBlock")
except Exception as e:
    print(f"✗ MineBlock 导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from basicsr.archs.lied_arch import LIED
    print("✓ 成功导入 LIED")
except Exception as e:
    print(f"✗ LIED 导入失败: {e}")
    import traceback
    traceback.print_exc()

def test_mineblock():
    """测试 MineBlock"""
    print("\n=== 测试 MineBlock ===")
    
    # 创建 MineBlock
    block = MineBlock(hidden_dim=32, channel_first=False)
    block.eval()
    
    # 测试输入
    x = torch.randn(2, 64, 64, 32)  # (B, H, W, C)
    
    try:
        with torch.no_grad():
            out = block(x)
        print(f"✓ MineBlock 前向传播成功")
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        
        # 计算参数量
        params = sum(p.numel() for p in block.parameters())
        print(f"  参数量: {params / 1e6:.3f}M")
        
    except Exception as e:
        print(f"✗ MineBlock 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

def test_lied():
    """测试 LIED 模型"""
    print("\n=== 测试 LIED 模型 ===")
    
    # 测试原始版本
    print("\n1. 原始 VSSLocalBlock 版本:")
    try:
        model_old = LIED(use_mineblock=False)
        model_old.eval()
        params_old = sum(p.numel() for p in model_old.parameters())
        print(f"  参数量: {params_old / 1e6:.2f}M")
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out_old = model_old(x)
        print(f"  ✓ 前向传播成功，输出形状: {out_old.shape}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试 MineBlock 版本
    print("\n2. MineBlock 版本:")
    try:
        model_new = LIED(use_mineblock=True)
        model_new.eval()
        params_new = sum(p.numel() for p in model_new.parameters())
        print(f"  参数量: {params_new / 1e6:.2f}M")
        
        if 'model_old' in locals():
            reduction = (params_old - params_new) / 1e6
            reduction_pct = (params_old - params_new) / params_old * 100
            print(f"  参数量减少: {reduction:.2f}M ({reduction_pct:.1f}%)")
        
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out_new = model_new(x)
        print(f"  ✓ 前向传播成功，输出形状: {out_new.shape}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mineblock()
    test_lied()
    print("\n=== 测试完成 ===")

