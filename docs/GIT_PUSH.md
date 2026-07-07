# Git 提交与推送到 GitHub

## 当前配置

```ini
# .git/config
[remote "origin"]
    url = https://github.com/fakesucker/hw_cosy.git
```

## 常见报错与原因

| 报错 | 原因 | 处理 |
|------|------|------|
| `Could not resolve host: github.com` | **计算节点无外网/DNS**（如 node37） | 换有网络的机器推送，见下文 |
| `Authentication failed` / `403` | HTTPS 未配置 Token | 使用 PAT，见「认证配置」 |
| `Permission denied (publickey)` | SSH 未配密钥 | 配置 SSH key，见「认证配置」 |
| `has no upstream branch` | 首次推送 | `git push -u origin main` |

> **node37 上的报错属于网络问题，不是仓库权限问题。** 先在能访问 GitHub 的环境完成 push。

---

## 方案 A：在有网络的机器上推送（推荐）

集群 **登录节点** 或 **本地电脑** 通常能访问外网：

```bash
cd /home/work_nfs23/hkxie/hw_proj/CosyVoice   # NFS 路径各节点一致

# 确认已有提交
git log -1 --oneline

# 首次推送
git push -u origin main

# 之后
git push
```

---

## 方案 B：计算节点无外网 — 用 bundle 中转

在 **node37（无外网）** 打包：

```bash
cd /home/work_nfs23/hkxie/hw_proj/CosyVoice
git bundle create /tmp/hw_cosy_main.bundle main
```

把 bundle 拷到能上网的机器（`scp` 到笔记本等），再：

```bash
git clone /path/to/hw_cosy_main.bundle hw_cosy
cd hw_cosy
git remote add origin https://github.com/fakesucker/hw_cosy.git
git push -u origin main
```

---

## 认证配置

### HTTPS + Personal Access Token（推荐）

1. GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens**
2. 仓库选 `fakesucker/hw_cosy`，权限至少 **Contents: Read and write**
3. 推送时：

```bash
git push -u origin main
# Username: fakesucker
# Password: <粘贴 Token，不是登录密码>
```

或写入 credential（仅本机）：

```bash
git config --global credential.helper store
# 首次 push 输入一次 Token 后会保存
```

### SSH（单账号）

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub   # 添加到 GitHub → Settings → SSH and keys
git remote set-url origin git@github.com:fakesucker/hw_cosy.git
git push -u origin main
```

### SSH 多账号：fakesucker + tiamojames（方案 C）

当 `ssh -T git@github.com` 显示 `Hi tiamojames!`，但仓库在 `fakesucker` 下时：

```bash
# 1. 为 fakesucker 单独生成密钥（不要覆盖现有 id_ed25519）
ssh-keygen -t ed25519 -C "fakesucker@github" -f ~/.ssh/id_ed25519_fakesucker

# 2. 配置 ~/.ssh/config（追加以下内容）
cat >> ~/.ssh/config << 'EOF'

Host github-fakesucker
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_fakesucker
    IdentitiesOnly yes

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

# 3. 把 fakesucker 公钥添加到 fakesucker 账号（不是 tiamojames）
cat ~/.ssh/id_ed25519_fakesucker.pub
# GitHub 登录 fakesucker → Settings → SSH and keys → New SSH key → 粘贴

# 4. 验证（应显示 Hi fakesucker!）
ssh -T git@github-fakesucker

# 5. 仓库 remote 使用别名 Host
cd /home/work_nfs23/hkxie/hw_proj/CosyVoice
git remote set-url origin git@github-fakesucker:fakesucker/hw_cosy.git
git push -u origin main
```

要点：`IdentityFile` 指向 fakesucker 专用密钥；`IdentitiesOnly yes` 避免 SSH 误用 tiamojames 的默认密钥。

---

## 代理（若集群要求走代理）

```bash
# 按实际代理地址修改
export https_proxy=http://proxy.example.com:8080
export http_proxy=http://proxy.example.com:8080

git push -u origin main
```

或仅对 GitHub：

```bash
git config --global http.https://github.com.proxy http://proxy.example.com:8080
```

---

## 推送前自检（可选）

```bash
# 网络
getent hosts github.com || echo "DNS 不通，请换节点或走代理"

# 远程
git remote -v

# 待推送提交
git log origin/main..HEAD --oneline 2>/dev/null || git log -1 --oneline
```

---

## 建议提交范围

`.gitignore` 已排除大文件与本地数据，正常只需提交代码与配置：

- `README.md`、`docs/`、`examples/huawei_sft/conf/`、`run_*.sh`
- `data_list/**/*.lst`（数据索引）
- **不要提交**：`*.pt`、`*.wav`、`testout/`、`data_list/**/*.jsonl`
