#!/bin/bash
# 系统配置检查脚本

echo "=========================================="
echo "系统配置检查"
echo "=========================================="

echo ""
echo "1. GPU 信息:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "未找到 nvidia-smi，可能没有 NVIDIA GPU 或驱动未安装"
fi

echo ""
echo "2. CUDA 版本:"
echo "----------------------------------------"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "未找到 nvcc，CUDA toolkit 可能未安装"
fi

echo ""
echo "3. Python 版本:"
echo "----------------------------------------"
python --version 2>/dev/null || python3 --version

echo ""
echo "4. Conda 信息:"
echo "----------------------------------------"
if command -v conda &> /dev/null; then
    conda --version
    echo "Conda 路径: $(which conda)"
    echo ""
    echo "当前激活的环境:"
    echo "$CONDA_DEFAULT_ENV"
else
    echo "未找到 conda"
fi

echo ""
echo "5. 已安装的 PyTorch (如果存在):"
echo "----------------------------------------"
python -c "
try:
    import torch
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'CUDA 可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'PyTorch CUDA 版本: {torch.version.cuda}')
        print(f'GPU 数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'  CUDA 计算能力: {torch.cuda.get_device_capability(i)}')
    else:
        print('CUDA 不可用（可能是 CPU 版本）')
except ImportError:
    print('PyTorch 未安装')
" 2>/dev/null || echo "无法检查 PyTorch（可能未安装）"

echo ""
echo "6. 系统信息:"
echo "----------------------------------------"
echo "操作系统: $(uname -a)"
echo "架构: $(uname -m)"

echo ""
echo "7. 环境变量:"
echo "----------------------------------------"
echo "CUDA_HOME: ${CUDA_HOME:-未设置}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-未设置}"
echo "PATH: ${PATH}"

echo ""
echo "=========================================="
echo "检查完成"
echo "=========================================="







