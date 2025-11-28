# Git 使用指南

## 1. 首次配置 Git（如果还没配置）

```bash
# 配置全局用户信息（所有仓库共用）
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱@example.com"

# 或者只为当前仓库配置
git config user.name "你的姓名"
git config user.email "你的邮箱@example.com"
```

## 2. 基本 Git 操作

### 查看状态
```bash
git status                    # 查看当前状态
git status --short           # 简短格式
```

### 添加文件
```bash
git add .                    # 添加所有文件
git add <文件名>             # 添加特定文件
git add basicsr/archs/       # 添加整个目录
```

### 提交更改
```bash
git commit -m "提交信息"     # 提交更改
git commit -am "提交信息"    # 添加并提交（仅限已跟踪文件）
```

### 查看历史
```bash
git log                      # 查看完整历史
git log --oneline           # 单行显示
git log --graph --oneline   # 图形化显示
```

## 3. 推送到远程仓库（GitHub/Gitee等）

### 方法1：GitHub

```bash
# 1. 在 GitHub 上创建新仓库（不要初始化 README）

# 2. 添加远程仓库
git remote add origin https://github.com/你的用户名/仓库名.git

# 3. 推送代码
git branch -M main           # 重命名分支为 main（可选）
git push -u origin main      # 首次推送
```

### 方法2：Gitee（码云）

```bash
# 1. 在 Gitee 上创建新仓库

# 2. 添加远程仓库
git remote add origin https://gitee.com/你的用户名/仓库名.git

# 3. 推送代码
git branch -M main
git push -u origin main
```

### 方法3：SSH 方式（推荐，更安全）

```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "你的邮箱@example.com"

# 2. 复制公钥到 GitHub/Gitee
cat ~/.ssh/id_ed25519.pub

# 3. 使用 SSH URL 添加远程仓库
git remote add origin git@github.com:你的用户名/仓库名.git
# 或
git remote add origin git@gitee.com:你的用户名/仓库名.git

# 4. 推送
git push -u origin main
```

## 4. 日常使用流程

```bash
# 1. 查看更改
git status

# 2. 添加更改
git add .

# 3. 提交更改
git commit -m "描述你的更改"

# 4. 推送到远程
git push
```

## 5. 常用命令

```bash
# 查看远程仓库
git remote -v

# 查看分支
git branch

# 创建新分支
git branch 新分支名

# 切换分支
git checkout 分支名

# 查看差异
git diff                    # 工作区 vs 暂存区
git diff --staged           # 暂存区 vs 上次提交

# 撤销更改
git restore <文件>          # 撤销工作区更改
git restore --staged <文件> # 取消暂存

# 查看特定文件的提交历史
git log -- <文件路径>
```

## 6. 当前仓库状态

✅ Git 仓库已初始化
✅ .gitignore 已创建
✅ 文件已添加到暂存区

**下一步：**
1. 配置用户信息（见步骤1）
2. 执行提交：`git commit -m "Initial commit"`
3. 添加远程仓库并推送（见步骤3）

