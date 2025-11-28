#!/bin/bash
# 恢复项目到 11 月 27 号状态的脚本

echo "=========================================="
echo "LIEDNet 项目恢复脚本"
echo "=========================================="
echo ""
echo "注意：此脚本将初始化 Git 仓库并创建当前状态的备份"
echo "然后您可以使用 Cursor 的 Git 功能查看历史记录"
echo ""
read -p "是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 1. 初始化 Git 仓库（如果还没有）
if [ ! -d ".git" ]; then
    echo "正在初始化 Git 仓库..."
    git init
    echo "✓ Git 仓库已初始化"
else
    echo "✓ Git 仓库已存在"
fi

# 2. 创建当前状态的备份
echo "正在创建当前状态的备份..."
git add .
git commit -m "Backup: Current state before restore to 2024-11-27 20:00" || echo "警告：提交失败（可能没有更改）"

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo ""
echo "现在您可以："
echo "1. 在 Cursor 中使用 Git 历史记录查看 11 月 27 号的版本"
echo "2. 或者使用 Cursor 的本地历史记录功能"
echo ""
echo "提示：在 Cursor 中，右键点击文件 → 'Local History' 查看文件历史"



