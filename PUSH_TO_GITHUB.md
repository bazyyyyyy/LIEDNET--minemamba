# 推送到 GitHub 指南

## ⚠️ 重要提示
GitHub 已经**不再支持密码认证**，需要使用 **Personal Access Token (PAT)** 来推送代码。

## 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub (https://github.com)
2. 点击右上角的 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `LIEDNet` (或你喜欢的名字)
   - Description: 可选
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"
4. 点击 "Create repository"

## 步骤 2: 创建 Personal Access Token (PAT)

1. 点击 GitHub 右上角头像 → **Settings**
2. 左侧菜单最下方 → **Developer settings**
3. 点击 **Personal access tokens** → **Tokens (classic)**
4. 点击 **Generate new token** → **Generate new token (classic)**
5. 填写信息：
   - Note: `LIEDNet Push Token` (描述性名称)
   - Expiration: 选择过期时间（建议 90 days 或 No expiration）
   - 勾选权限：至少需要 `repo` 权限（全选 repo 下的所有权限）
6. 点击 **Generate token**
7. **重要**：复制生成的 token（只显示一次！），格式类似：`ghp_xxxxxxxxxxxxxxxxxxxx`

## 步骤 3: 添加远程仓库并推送

在终端执行以下命令：

```bash
cd /home/bazy/下载/LIEDNet

# 1. 添加远程仓库（替换为你的仓库URL）
git remote add origin https://github.com/bazyyyyy/LIEDNet.git

# 2. 重命名分支为 main（GitHub 默认使用 main）
git branch -M main

# 3. 推送代码（会提示输入用户名和密码）
# 用户名：bazyyyyy
# 密码：输入刚才复制的 Personal Access Token（不是 GitHub 密码！）
git push -u origin main
```

## 方法 2: 使用 SSH（推荐，更安全）

### 2.1 生成 SSH 密钥

```bash
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "bazyyyyy@users.noreply.github.com"

# 按 Enter 使用默认路径
# 可以设置密码（可选，更安全）

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

### 2.2 添加 SSH 密钥到 GitHub

1. 复制刚才显示的公钥内容
2. GitHub → Settings → SSH and GPG keys
3. 点击 "New SSH key"
4. Title: `My Linux PC` (描述性名称)
5. Key: 粘贴公钥内容
6. 点击 "Add SSH key"

### 2.3 使用 SSH 推送

```bash
# 使用 SSH URL 添加远程仓库
git remote set-url origin git@github.com:bazyyyyy/LIEDNet.git

# 或如果还没添加远程仓库
git remote add origin git@github.com:bazyyyyy/LIEDNet.git

# 推送
git push -u origin main
```

## 日常使用

推送新更改：

```bash
git add .
git commit -m "描述你的更改"
git push
```

## 如果遇到问题

### 问题1: 认证失败
- 确保使用 Personal Access Token 而不是密码
- 检查 token 是否过期
- 检查 token 是否有 `repo` 权限

### 问题2: 远程仓库已存在内容
```bash
# 如果远程仓库有 README 等文件，需要先拉取
git pull origin main --allow-unrelated-histories
# 解决冲突后
git push -u origin main
```

### 问题3: 查看远程仓库
```bash
git remote -v
```

### 问题4: 更改远程仓库 URL
```bash
git remote set-url origin <新的URL>
```

