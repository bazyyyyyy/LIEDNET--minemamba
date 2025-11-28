#!/bin/bash
# LIEDNet 环境快速安装脚本

echo "=========================================="
echo "LIEDNet 环境安装脚本"
echo "=========================================="

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 1. 创建环境
echo "步骤 1: 创建 conda 环境..."
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
else
    echo "未找到 environment.yml，使用基础安装..."
    conda create -n liednet python=3.10.19 -y
    conda activate liednet
    pip install -r requirements.txt
fi

# 2. 激活环境
echo "步骤 2: 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate liednet

# 3. 检查并安装 PyTorch
echo "步骤 3: 检查 PyTorch..."
python -c "import torch" 2>/dev/null || {
    echo "PyTorch 未安装，正在安装..."
    pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu118
}

# 4. 安装额外依赖
echo "步骤 4: 安装额外依赖..."
pip install einops fvcore ptflops tensorboardx timm

# 5. 编译 selective_scan
echo "步骤 5: 编译 selective_scan CUDA 扩展..."
if [ -d "kernels/selective_scan" ]; then
    cd kernels/selective_scan
    pip install -e .
    cd ../..
    echo "selective_scan 编译完成"
else
    echo "警告: 未找到 kernels/selective_scan 目录"
fi

# 6. 验证安装
echo "步骤 6: 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

python -c "
try:
    import selective_scan_cuda_core
    print('✓ selective_scan CUDA extension: OK')
except:
    print('⚠ selective_scan CUDA extension: Not found (will use fallback)')
"

echo "=========================================="
echo "安装完成！"
echo "请运行: conda activate liednet"
echo "=========================================="
