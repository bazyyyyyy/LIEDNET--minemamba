#!/bin/bash

echo "=========================================="
echo "LIEDNet 环境安装"
echo "=========================================="

# 步骤 1: 创建环境
echo ""
echo "步骤 1: 创建 conda 环境..."
conda create -n liednet python=3.10.19 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate liednet

# 步骤 2: 安装 PyTorch
echo ""
echo "步骤 2: 安装 PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
pip install torch torchvision)

# 步骤 3: 安装基础依赖
echo ""
echo "步骤 3: 安装基础依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "警告: 未找到 requirements.txt"
fi

# 步骤 4: 安装额外依赖
echo ""
echo "步骤 4: 安装额外依赖..."
pip install einops fvcore ptflops tensorboardx timm

# 步骤 5: 编译 selective_scan
echo ""
echo "步骤 5: 编译 selective_scan CUDA 扩展..."
if [ -d "kernels/selective_scan" ]; then
    cd kernels/selective_scan
    pip install -e .
    cd ../..
    echo "✓ selective_scan 编译完成"
else
    echo "⚠ 警告: 未找到 kernels/selective_scan 目录"
fi

# 步骤 6: 验证
echo ""
echo "步骤 6: 验证安装..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "PyTorch 验证失败"

python -c "import einops, fvcore, ptflops, tensorboardx, timm; print('✓ 所有依赖安装成功')" 2>/dev/null || echo "⚠ 部分依赖缺失"

python -c "try:
    import selective_scan_cuda_core
    print('✓ selective_scan CUDA 扩展: 安装成功')
except:
    print('⚠ selective_scan CUDA 扩展: 未找到（将使用 fallback）')" 2>/dev/null

echo ""
echo "=========================================="
echo "安装完成！"
echo ""
echo "下一步："
echo "1. 运行: conda activate liednet"
echo "2. 修改 options/train_LOLv1.yaml 中的数据集路径"
echo "3. 运行: python train_llie.py --opt ./options/train_LOLv1.yaml"
echo "=========================================="
