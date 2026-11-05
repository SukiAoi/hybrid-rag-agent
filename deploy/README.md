# 部署清单（首尔 2C4G · Ubuntu 22.04）

> 目标：`https://你的域名` 能打开并回答一个问题。
> 全程约 40 分钟，其中 20 分钟是等 DNS 和证书。

## 0. 前置

- [ ] 服务器已开机，拿到公网 IP
- [ ] 域名已买（`.top` / `.xyz` 十几块）
- [ ] 域名 A 记录 → 服务器 IP（**首尔不需要备案**，解析直接生效）

## 1. 服务器初始化

SSH 登录后依次执行：

```bash
# 1.1 更新 + 基础工具
apt update && apt upgrade -y
apt install -y curl git ufw

# 1.2 建一个普通用户，别用 root 跑服务
adduser --disabled-password --gecos "" deploy
usermod -aG sudo deploy
mkdir -p /home/deploy/.ssh && cp ~/.ssh/authorized_keys /home/deploy/.ssh/ 2>/dev/null
chown -R deploy:deploy /home/deploy/.ssh && chmod 700 /home/deploy/.ssh

# 1.3 加 2G swap（2C4G 也建议加，防构建时 OOM）
fallocate -l 2G /swapfile && chmod 600 /swapfile
mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# 1.4 防火墙：只开 22 / 80 / 443
ufw allow 22 && ufw allow 80 && ufw allow 443
ufw --force enable

# 1.5 装 Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker deploy
```

> 之后用 `deploy` 用户操作，`sudo` 提权。

## 2. 放配置

```bash
sudo mkdir -p /opt/rag && sudo chown deploy:deploy /opt/rag
cd /opt/rag
```

把仓库里 `deploy/` 下三个文件传上去（用 scp 或直接 `git clone` 整个仓库再拷）：

```bash
# 本地执行
scp deploy/docker-compose.prod.yml deploy/Caddyfile deploy/.env.prod.example deploy@<IP>:/opt/rag/
```

```bash
# 服务器上
cd /opt/rag
mv .env.prod.example .env
nano .env          # 填 DEEPSEEK_API_KEY 和 DOMAIN
```

## 3. 起服务

```bash
cd /opt/rag
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs -f caddy
```

Caddy 会自动向 Let's Encrypt 申请证书，看到 `certificate obtained successfully` 就成了。

首次拉镜像约 300MB，加上 ONNX 模型下载（79MB），大概 2–4 分钟。

## 4. 验收

```bash
# 4.1 健康检查（本地）
curl http://127.0.0.1:8000/health
# 期望 {"status":"ok","kb_chunks":13}

# 4.2 HTTPS + 证书
curl -I https://你的域名/
# 期望 HTTP/2 200，且没有证书警告

# 4.3 真实问答
curl -s https://你的域名/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"玩家移动状态机有哪些状态？"}' | head -c 500
```

**四件事都过了，M1 就算完成。**

## 5. 接 CI/CD（可选但强烈建议）

仓库 → Settings → Secrets and variables → Actions，加三个：

| Secret | 值 |
|---|---|
| `SSH_HOST` | 服务器 IP |
| `SSH_USER` | `deploy` |
| `SSH_KEY` | 私钥全文（`-----BEGIN OPENSSH PRIVATE KEY-----` 开头那整段） |

配好之后，每次 push 到 main：跑测试 → 构建镜像 → 推 ghcr → SSH 到服务器拉取重启。

## 6. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 证书申请失败 | DNS 还没生效 / 80 端口被占 | `dig 你的域名` 确认；`ss -lntp \| grep 80` |
| 502 | api 容器没起来 | `docker compose logs api` |
| 打开很慢（>5s） | 首尔 + 移动晚高峰 | 正常，先接受；后续迁香港 |
| 内存告警 | Chroma + ONNX 超了 | `docker stats`，把 api 的 `mem_limit` 调到 1500m |
| `kb_chunks: 0` | 知识库没建好 | `docker compose logs api \| grep 知识库` |

## 7. 上线后立刻做

- [ ] 简历三版都加上 `Demo: https://你的域名`
- [ ] GitHub 仓库 README 顶部加一行 demo 链接
- [ ] 截图一张（浏览器 + 地址栏），放进 README
- [ ] 在 Boss 打招呼话术里加一句"可在线体验"

---

## 安全边界（被问到要能答）

现在做到的：HTTPS（Caddy 自动续期）、非 root 容器、只开三个端口、请求超时、入参长度限制、容器日志轮转。

**还没做的**（别在面试里说做了）：限流、鉴权、只读文件系统、seccomp、WAF、监控告警。

下一步优先级：限流 > 成本看板 > 鉴权。
