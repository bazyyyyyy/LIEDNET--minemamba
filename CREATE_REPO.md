# 在 GitHub 创建仓库

## ✅ SSH 连接已成功！

你的 SSH 密钥已正确配置，可以连接到 GitHub。

## 📝 下一步：创建 GitHub 仓库

### 步骤 1: 创建新仓库

1. 访问：https://github.com/new
2. **Repository name**: `LIEDNet`（或你喜欢的名字）
3. **Description**: 可选，例如 "Low-light Image Enhancement with MineBlock"
4. **Visibility**: 选择 Public 或 Private
5. ⚠️ **重要**：**不要**勾选以下选项：
   - ❌ Add a README file
   - ❌ Add .gitignore
   - ❌ Choose a license
6. 点击 **"Create repository"**

### 步骤 2: 推送代码

创建仓库后，执行：

```bash
cd /home/bazy/下载/LIEDNet
git push -u origin main
```

## 🎉 完成！

推送成功后，你的代码就会出现在 GitHub 上：
https://github.com/bazyyyyy/LIEDNet

## 如果仓库名不同

如果创建的仓库名不是 `LIEDNet`，需要更新远程 URL：

```bash
git remote set-url origin git@github.com:bazyyyyy/你的仓库名.git
git push -u origin main
```

