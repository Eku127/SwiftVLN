# 98服务器 SSH 隧道代理修复方案

## 一、问题描述

98服务器（`10.246.152.98`）需要通过本机的 Clash 代理（端口 7890）访问外网（OpenAI 等），但每当 VPN1（连接服务器的 VPN）重连后，服务器就无法访问外网。

## 二、网络架构

```
本机 ←── VPN1 ──→ 98服务器 (10.246.152.98)
  │
  └── Clash代理 (127.0.0.1:7890) ──→ 外网 (OpenAI)

98服务器 ──→ SSH RemoteForward:7890 ──→ 本机 Clash:7890 ──→ 外网
```

- 本机通过 VPN1 连接 98 服务器
- 本机运行 Clash 代理（端口 7890）提供外网访问
- SSH 的 `RemoteForward` 将本机的 7890 端口映射到服务器的 7890 端口
- 服务器上的程序通过 `http://127.0.0.1:7890` 代理访问外网

## 三、根因分析

### 故障链

```
VPN1 重连
  → SSH 连接瞬间断开
  → 服务器 sshd 进程未感知连接已断（TCP 未收到 FIN/RST）
  → 旧 sshd 进程继续占用 7890 端口（僵尸监听）
  → VPN1 恢复后新 SSH 连接建立
  → RemoteForward 7890 发现端口被占用 → 静默失败
  → 服务器 7890 端口虽在监听，但实际是僵尸隧道，流量无法转发
  → 服务器断网
```

### 核心原因

1. **服务器 sshd 默认不检测客户端存活**（`ClientAliveInterval` 默认为 0）
2. **僵尸 sshd 进程由 root 持有**，普通用户无法杀掉
3. **新连接的 RemoteForward 端口冲突时静默失败**，不会报错

## 四、解决方案

### 方案总览

| 步骤 | 措施 | 作用 | 权限要求 |
|------|------|------|----------|
| 1 | 服务端 sshd 心跳检测 | 自动清除死连接，释放端口 | **sudo** |
| 2 | SSH config 分离隧道 | 避免多连接抢占端口 | 普通用户 |
| 3 | 守护脚本自动重连 | 隧道断了自动恢复 | 普通用户 |
| 4 | 开机自启 | 每次登录自动运行守护脚本 | 普通用户 |

---

### 步骤 1：服务端 sshd 心跳检测（最关键，需 sudo）

**在 98 服务器上执行：**

```bash
sudo sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 15/' /etc/ssh/sshd_config
sudo sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 3/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

**验证：**

```bash
grep ClientAlive /etc/ssh/sshd_config
# 应输出：
# ClientAliveInterval 15
# ClientAliveCountMax 3
```

**作用：** 服务端每 15 秒向客户端发送心跳，连续 3 次无响应（45 秒）自动断开死连接并释放端口。这是唯一能彻底解决僵尸端口问题的方法。

---

### 步骤 2：修改本机 SSH Config

**文件路径：** `C:\Users\LASER\.ssh\config`

**修改内容：** 移除 Cursor 连接中的 `RemoteForward`，只保留保活参数。端口转发由独立的守护脚本负责。

```
Host H100-develop-2
  HostName 10.246.152.98
  User develop
  ServerAliveInterval 30
  ServerAliveCountMax 3

Host H100-jiangjiajun-2
  HostName 10.246.152.98
  User jiangjiajun
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

**作用：**
- 移除 `RemoteForward 7890`：Cursor 的多个 SSH 连接不再各自抢占 7890 端口
- `ServerAliveInterval 30`：客户端每 30 秒检测一次服务器存活
- `ServerAliveCountMax 3`：连续 3 次无响应后断开（90 秒）

---

### 步骤 3：部署守护脚本

**文件路径：** `C:\Users\LASER\ssh-tunnel-guard.ps1`

**脚本功能：**
- 建立独立的 SSH 隧道连接（仅做 RemoteForward，不执行命令）
- 每 10 秒检测隧道是否存活
- 隧道断开后自动尝试清理远程僵尸端口并重连
- 使用 `ExitOnForwardFailure=yes` 确保端口绑定失败时立即退出触发重试
- SSH 心跳间隔 15 秒，最多容忍 3 次失败（45 秒检测到断连）

**手动运行：**

```powershell
powershell -ExecutionPolicy Bypass -File C:\Users\LASER\ssh-tunnel-guard.ps1
```

---

### 步骤 4：设置开机自启

已通过 Windows 任务计划程序配置：

| 配置项 | 值 |
|--------|------|
| 任务名称 | `SSH-Tunnel-Guard` |
| 触发条件 | 用户 LASER 登录时 |
| 运行方式 | 后台隐藏窗口 |
| 执行命令 | `powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File C:\Users\LASER\ssh-tunnel-guard.ps1` |

**管理命令：**

```powershell
# 查看状态
Get-ScheduledTask -TaskName "SSH-Tunnel-Guard"

# 手动启动
Start-ScheduledTask -TaskName "SSH-Tunnel-Guard"

# 停止
Stop-ScheduledTask -TaskName "SSH-Tunnel-Guard"

# 删除
Unregister-ScheduledTask -TaskName "SSH-Tunnel-Guard"
```

## 五、VPN 断连后的恢复流程

### 配置了 sshd ClientAliveInterval 后（自动恢复）

```
VPN1 断连
  → 服务器 45 秒内检测到死连接
  → 自动杀掉旧 sshd，释放 7890 端口
  → VPN1 恢复
  → 守护脚本检测到隧道断开
  → 自动重连，RemoteForward 成功绑定 7890
  → 服务器恢复外网访问
  （全程自动，无需干预）
```

### 未配置 sshd ClientAliveInterval（需手动干预）

```
VPN1 断连并重连后，执行以下命令手动清理：

# 1. 找出旧的 sshd 会话（看时间戳，早于 VPN 重连时间的就是旧的）
ssh H100-jiangjiajun-2 "ps aux | grep 'sshd.*jiangjiajun' | grep -v grep"

# 2. 杀掉旧的 @notty 会话（替换为实际 PID）
ssh H100-jiangjiajun-2 "kill <旧PID1> <旧PID2> ..."

# 3. 确认端口已释放
ssh H100-jiangjiajun-2 "ss -tlnp | grep 7890 || echo '端口已释放'"

# 4. 等待守护脚本自动重建隧道（约 10-20 秒），或手动建立：
ssh -N -o ExitOnForwardFailure=yes -R 7890:127.0.0.1:7890 H100-jiangjiajun-2
```

## 六、验证与排查

### 验证连通性

```bash
# 在 98 服务器上执行
https_proxy=http://127.0.0.1:7890 curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://api.openai.com/v1/models -H 'Authorization: Bearer test-key'
# 返回 401 = 网络正常（API Key 为测试值）
```

### 检查端口状态

```bash
# 在 98 服务器上执行
ss -tlnp | grep 7890
# 有 LISTEN 输出 = 端口已绑定
```

### 检查代理是否真正可用

```bash
# 在 98 服务器上执行
https_proxy=http://127.0.0.1:7890 curl -v --max-time 10 https://www.google.com 2>&1 | head -20
# 有 HTTP 响应 = 代理正常；超时 = 僵尸隧道需清理
```

### 查看守护脚本是否运行

```powershell
# 在本机执行
Get-ScheduledTask -TaskName "SSH-Tunnel-Guard" | Select-Object State
Get-Process -Name ssh -ErrorAction SilentlyContinue
```

## 七、相关文件

| 文件 | 路径 | 说明 |
|------|------|------|
| SSH Config | `C:\Users\LASER\.ssh\config` | SSH 连接配置 |
| 守护脚本 | `C:\Users\LASER\ssh-tunnel-guard.ps1` | 隧道自动重连 |
| 任务计划 | `SSH-Tunnel-Guard` | 开机自启任务 |
| 服务端 sshd 配置 | `/etc/ssh/sshd_config`（98 服务器） | 心跳检测配置 |
| codex 配置 | `~/.codex/config.toml`（98 服务器） | codex 设置 |
| codex 认证 | `~/.codex/auth.json`（98 服务器） | ChatGPT 认证 |
