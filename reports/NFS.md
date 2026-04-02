# 双机 NFS 代码目录共享配置文档
**创建日期**：2025-12-29  
**目标**：实现从主控节点 (.98) 到计算节点 (.73) 的代码实时共享，确保代码维护单一来源，无需手动复制。

## 1. 环境拓扑信息
| 角色 | 节点名称 | IP 地址 | 职责 |
| --- | --- | --- | --- |
| **Server (服务端)** | **Master** | `10.246.152.98` | 代码开发、存储核心数据、NFS 共享源 |
| **Client (客户端)** | **Worker** | `10.246.152.73` | 挂载共享目录、执行训练/推理任务 |


+ **共享路径 (双方一致)**：`/mnt/data1/home/jiangjiajun/workspace`
+ **网络环境**：普通内网连接 (非 RDMA)

---

## 2. 服务端配置 (在 .98 上执行)
**步骤 2.1：安装 NFS 服务组件**

```bash
sudo apt update
sudo apt install nfs-kernel-server -y

```

**步骤 2.2：配置共享策略**  
编辑 NFS 导出配置文件：

```bash
sudo vim /etc/exports

```

在文件末尾添加以下配置（指定仅允许 .73 访问，权限为读写）：

```latex
/mnt/data1/home/jiangjiajun/workspace 10.246.152.73(rw,sync,no_subtree_check,no_root_squash)

```

> **参数释义**：
>
> + `rw`: 读写权限 (Read/Write)。
> + `sync`: 数据实时同步写盘（保证安全性）。
> + `no_root_squash`: 允许客户端以 root 身份操作文件（避免权限不足问题）。
>
> 
>

**步骤 2.3：重启服务生效**

```bash
# 重新加载配置
sudo exportfs -a

# 重启 NFS 服务
sudo systemctl restart nfs-kernel-server

```

---

## 3. 客户端配置 (在 .73 上执行)
**步骤 3.1：安装 NFS 客户端组件**

```bash
sudo apt update
sudo apt install nfs-common -y

```

_(注：安装过程中若出现粉色弹窗“Services to restart”，保持默认直接回车即可)_

**步骤 3.2：创建挂载点**  
确保本地存在同名目录（作为挂载的入口）：

```bash
mkdir -p /mnt/data1/home/jiangjiajun/workspace

```

**步骤 3.3：执行手动挂载**  
将远程目录挂载到本地：

```bash
sudo mount 10.246.152.98:/mnt/data1/home/jiangjiajun/workspace /mnt/data1/home/jiangjiajun/workspace

```

**步骤 3.4：验证挂载状态**  
查看目录内容，应能看到 .98 上的文件（如 `ms-swift`）：

```bash
ls -l /mnt/data1/home/jiangjiajun/workspace
# 或者使用 df -h 查看挂载点容量信息
df -h | grep 98

```

---

## 4. 设置开机自动挂载 (强烈推荐)
如果不进行此步，机器重启后挂载会失效。我们需要修改 `/etc/fstab` 文件。

**在 .73 (Client) 上执行：**

1. 编辑 fstab 文件：

```bash
sudo vim /etc/fstab

```



2. 在文件最后一行添加：

```latex
10.246.152.98:/mnt/data1/home/jiangjiajun/workspace /mnt/data1/home/jiangjiajun/workspace nfs defaults,_netdev 0 0

```



_(注：_`_netdev`_ 表示等到网络连接建立后再挂载，防止开机卡死)_  
3. 测试配置是否正确（**重要**）：

```bash
sudo mount -a

```



_如果没有报错，说明配置正确。_

---

## 5. 进阶：Conda 环境自动同步策略
为了避免在两台机器重复安装 Python 包，建议将 Conda 环境直接安装在 NFS 共享目录下。

**操作位置**：只在 **.98 (Server)** 上操作，.73 即可直接使用。

1. **创建存放环境的目录**：

```bash
mkdir -p /mnt/data1/home/jiangjiajun/workspace/envs

```



2. **创建环境 (指定路径)**：

```bash
# 示例：创建一个名为 shared_env 的环境
conda create --prefix /mnt/data1/home/jiangjiajun/workspace/envs/shared_env python=3.10 -y

```



3. **激活环境**：
+ 在 .98 上：`conda activate /mnt/data1/home/jiangjiajun/workspace/envs/shared_env`
+ 在 .73 上：`conda activate /mnt/data1/home/jiangjiajun/workspace/envs/shared_env`



---

## 6. 注意事项与性能警告
1. **数据存放原则**：
+ ✅ **代码、脚本、配置文件、轻量级模型 Checkpoint** -> 放在 **NFS 共享目录**。
+ ❌ **大规模训练数据集 (ImageNet, CommonCrawl 等)** -> **严禁**通过 NFS 读取。必须使用 `rsync` 拷贝到 .73 的本地 NVMe 硬盘，否则 GPU 利用率会极低。



2. **文件一致性**：
+ 在 .98 修改代码后，.73 是毫秒级可见的。
+ 在 .73 训练生成的 log/checkpoint，会实时回写到 .98 的硬盘上。



3. **取消挂载 (运维用)**：  
如果需要停止共享，在 .73 上执行：

```bash
sudo umount /mnt/data1/home/jiangjiajun/workspace

```

这份文档详细记录了如何将 **.98 (Master)** 上的 Conda 环境通过 NFS 共享给 **.73 (Worker)**，实现环境的“一次配置，多机可用”。

建议将此文档保存为 `README_CONDA_SETUP.md`。

---

# 双机 Conda 环境共享配置文档
**创建日期**：2025-12-29  
**目标**：将 Master 节点的 Conda 安装目录完整挂载到 Worker 节点，使 Worker 能够直接调用 Master 上已安装的所有 Python 环境（如 `ewm`, `swift-vln`），无需重复安装依赖。

## 1. 基础信息
+ **Server (服务端)**: `10.246.152.98`
+ **Client (客户端)**: `10.246.152.73`
+ **Conda 基础路径**: `/mnt/data1/home/jiangjiajun/miniconda3`
+ _(注：此路径包含了 _`bin/conda`_ 可执行文件以及 _`envs/`_ 下的所有环境)_

---

## 2. 确认 Conda 安装位置 (在 .98 上执行)
在配置共享前，必须确保路径准确。

+ **方法**：运行以下命令查看 `base` 环境路径。

```bash
conda info
# 或者
conda env list

```

+ **确认**：根据你的系统情况，核心路径确认为 `/mnt/data1/home/jiangjiajun/miniconda3`。

---

## 3. 服务端配置 (在 .98 上执行)
**步骤 3.1：配置 NFS 共享**  
我们需要将整个 miniconda3 目录共享出去。

1. 编辑 NFS 配置文件：

```bash
sudo vim /etc/exports
```

2. **追加**以下配置行（保留之前的 workspace 配置，新增一行）：

```latex
/mnt/data1/home/jiangjiajun/miniconda3 10.246.152.73(rw,sync,no_subtree_check,no_root_squash)
```



**步骤 3.2：生效配置**

```bash
sudo exportfs -a
# 检查是否生效
sudo exportfs -v
```

---

## 4. 客户端配置 (在 .73 上执行)
**步骤 4.1：创建挂载点**  
为了保持路径一致性，必须在 .73 上创建一模一样的目录结构。

```bash
mkdir -p /mnt/data1/home/jiangjiajun/miniconda3

```

**步骤 4.2：执行挂载**  
将远程的 Conda 目录“投射”到本地：

```bash
sudo mount 10.246.152.98:/mnt/data1/home/jiangjiajun/miniconda3 /mnt/data1/home/jiangjiajun/miniconda3

```

**步骤 4.3：验证挂载**  
查看能否读取到环境列表：

```bash
ls -l /mnt/data1/home/jiangjiajun/miniconda3/envs

```

_如果你能看到 _`ewm`_, _`swift-vln`_ 等文件夹，说明挂载成功。_

---

## 5. 初始化 Conda (在 .73 上执行)
虽然文件已经有了，但 .73 的终端还需要“注册” `conda` 命令。

**步骤 5.1：初始化 Shell**  
使用挂载过来的二进制文件进行初始化：

```bash
# 执行初始化
/mnt/data1/home/jiangjiajun/miniconda3/bin/conda init bash

# 刷新环境变量
source ~/.bashrc

```

**步骤 5.2：最终测试**  
输入以下命令，检查是否显示了 98 上的所有环境：

```bash
conda env list

```

_预期输出：应该能看到 _`base`_, _`ewm`_, _`swift-vln`_ 等环境，且路径指向 _`/mnt/data1/...`_。_

---

## 6. 设置开机自动挂载 (防止重启失效)
**操作位置**：.73 (Client)

1. 编辑 fstab 文件：

```bash
sudo vim /etc/fstab
```



2. **追加**以下内容（建议将 workspace 和 miniconda3 都加上）：

```latex
# Workspace 代码目录
10.246.152.98:/mnt/data1/home/jiangjiajun/workspace /mnt/data1/home/jiangjiajun/workspace nfs defaults,_netdev 0 0

# Conda 环境目录
10.246.152.98:/mnt/data1/home/jiangjiajun/miniconda3 /mnt/data1/home/jiangjiajun/miniconda3 nfs defaults,_netdev 0 0
```



---

## 7. 日常使用与维护指南 (重要)
由于两台机器共用同一份文件，请遵循以下原则：

1. **安装包/创建环境**：
+ 👋 **请只在 .98 (Master) 上操作！**
+ 例如：想安装 `pandas`，请在 .98 上激活环境并 `pip install pandas`。
+ 原因：虽然 .73 也可以写入，但为了管理混乱，建议统一由 .98 进行环境管理。.98 装完，.73 秒级可见。



2. **运行代码**：
+ 👋 **在 .73 (Worker) 上直接运行。**
+ 例如：`conda activate swift-vln` -> `python train.py`。
+ 此时 Python 解释器和库文件都是通过网络从 .98 读取加载到 .73 的内存中运行。



3. **如果 .73 报错 "Text file busy"**：
+ 这种情况极少发生。如果 .98 正在更新某个 Python 包，而 .73 正在运行代码使用这个包，可能会报错。
+ **解决**：等 .98 安装完包，再在 .73 重新运行程序。

