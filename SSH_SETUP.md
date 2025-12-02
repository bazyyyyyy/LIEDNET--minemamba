# SSH 方式推送 GitHub（推荐）

## 步骤 1: 添加 SSH 密钥到 GitHub

你的 SSH 公钥：
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDX3xtH1hkJUjZHF6ohdlmp7U0Tyq93QJyls2xjg9qA/ bazyyyyy@users.noreply.github.com
```

操作步骤：
1. 复制上面的整个公钥（从 `ssh-ed25519` 到 `bazyyyyy@users.noreply.github.com`）
2. 访问：https://github.com/settings/keys
3. 点击 "New SSH key"
4. Title: `My Linux PC` (或任意描述)
5. Key: 粘贴刚才复制的公钥
6. 点击 "Add SSH key"

## 步骤 2: 更改远程仓库为 SSH

执行以下命令：

```bash
cd /home/bazy/下载/LIEDNet
git remote set-url origin git@github.com:bazyyyyy/LIEDNet.git
git branch -M main
git push -u origin main
```

## 优势

- ✅ 更稳定，不受 HTTPS 端口限制
- ✅ 更安全，不需要在 URL 中暴露 token
- ✅ 一次配置，永久使用

