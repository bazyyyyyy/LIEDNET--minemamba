#!/bin/bash
# 使用 Personal Access Token 推送代码到 GitHub

echo "正在推送代码到 GitHub..."
echo ""

# 设置远程仓库（如果还没设置）
git remote set-url origin https://github.com/bazyyyyy/LIEDNet.git

# 重命名分支为 main
git branch -M main

# 推送代码
# 当提示输入密码时，输入 Personal Access Token
echo "提示：当要求输入密码时，请输入你的 Personal Access Token"
echo "注意：Token 需要保存在安全的地方，不要提交到代码库"
echo ""
git push -u origin main

echo ""
echo "如果推送成功，你的代码已经上传到 GitHub！"
echo "如果遇到网络问题，请稍后重试或检查网络连接。"

