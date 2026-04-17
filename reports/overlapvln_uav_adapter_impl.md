# OverlapVLN UAV Adapter 实施文档（V1）

## 1. 文档目的

这份文档用于替代 [adapter.md](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/reports/adapter.md) 作为后续实现参考。

本版结论已经固定：

- **不做 token distill**
- 先做 **真实 UAV -> Satellite teacher space** 的视觉对齐
- 新模块放在 `src/` 下，而不是塞进 `reports/` 或临时脚本
- 目标不是单独做一个检索模型，而是产出一个后续能接入 **OverlapVLN** 的 `uav_adapter`

---

## 2. 当前仓库状态下的核心判断

### 2.1 当前最重要的事实

- SwiftVLN 当前主线模型是 `overlapvln`
- OverlapVLN 已经有一个现成的视觉增强插槽：
  - 视觉 encoder 输出 token 后
  - history 压缩前
  - 可插入 trainable enhancement module
- 对应代码位置：
  - [src/swiftvln/model/template.py](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/model/template.py#L573)
  - [src/swiftvln/model/template.py](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/model/template.py#L582)
  - [src/swiftvln/model/trainer.py](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/model/trainer.py#L99)
- 当前 OverlapVLN 训练默认并**没有 freeze ViT**：
  - [src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/model/script/train/train_overlapvln_qwen2_5_vl.sh#L168)

### 2.2 这直接带来的实现判断

- 如果最终目标是兼容**当前已经训练好的 OverlapVLN policy**，那么 stage1 对齐时的 teacher 不应该默认是原始 `Qwen2.5-VL-3B-Instruct`。
- 更合理的 teacher 是：
  - **目标 OverlapVLN checkpoint 内部的 visual tower 输出空间**
- 也就是说，V1 要对齐到“当前 policy 真正使用的视觉空间”，而不是只对齐到 base Qwen 的原始视觉空间。

### 2.3 当前建议的 teacher

先使用当前表现最强的 SatNav teacher 作为默认目标 policy：

- `output/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-noembed-data260317-bs64-lr2e-5-20260318-202149/v0-20260318-202212/checkpoint-3957`

该 checkpoint 的离线评测表现：

- `val_seen SR = 0.7334`
- `val_unseen SR = 0.6328`

对应汇总：

- [val_seen summary](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/eval/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-noembed-data260317-bs64-lr2e-5-20260318-202149/val_seen/20260329_065814/evaluation_summary.json#L3)
- [val_unseen summary](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/results/eval/overlapvln/overlapvln-satnav-stage1-3b-1ep-f32s4-overlap16-gtc-k512-noembed-data260317-bs64-lr2e-5-20260318-202149/val_unseen/20260329_075748/evaluation_summary.json#L3)

---

## 3. 数据现状与 V1 使用范围

### 3.1 实际数据路径

后续实现以这个目录为准：

- `/mnt/data3/jiangjiajun/dataset/SatDronePair`

不是：

- `/mnt/data3/home/jiangjiajun/satdronpair`

### 3.2 数据组成

当前共有约 `24,467` 对配对图像：

- `denseuav`: `5,464`
- `gta`: `10,204`
- `sues`: `1,497`
- `uavvisloc`: `7,302`

### 3.3 V1 的数据使用原则

- 只使用各数据集导出的 `drone/` 与 `satellite/` 成品图，不重新回到原始裁剪脚本链路
- 默认信任这些成品图已经做过 north-up 或等价朝向校正
- V1 不额外做人为 token 级几何对齐
- V1 不强求固定分辨率一致，交给 Qwen processor 处理动态 patch

### 3.4 V1 的 split 策略

统一 split 原则：

- **不直接沿用任何数据集原始的 `train/test` 字段**
- 四个数据源全部先视为 **可用训练样本池**
- 由我们统一重新生成 `train/val` 划分
- V1 暂时**不单独设 test split**
- `val` 只用于：
  - early stopping
  - 超参选择
  - 检索/对齐效果对比
- 不做逐样本随机切分，必须按场景或区域分组切分

具体建议：

- `denseuav`: 忽略原始 `split` 字段，按去掉高度后缀的基础位置 ID 分组切分；同一位置的 `H80/H90/H100` 必须进同一 split
- `gta`: 忽略原始 `split` 字段，按 `area_mode + satellite_img_name` 分组切分；同一 satellite tile 及其邻近重复样本不要同时落入 train 和 val
- `sues`: 按 `scene_id` 分组切分
- `uavvisloc`: 按 `seq_id` 分组切分

默认建议：

- 每个数据源单独做 `90/10` 的 `train/val`
- 若某个数据源分组数很少，则至少保留 `1` 个完整 group 到 `val`
- manifest 中写入我们重建后的 `split`，不要回写原始 CSV

这样做的目的不是做正式 benchmark，而是先得到一个**可信的开发验证集**，避免因为同区域泄漏导致 alignment 指标虚高。

---

## 4. V1 明确不做的事情

### 4.1 不做 token distill

原因不是“理论上不行”，而是当前工程目标下**性价比差**：

- 四个数据源的高度、视场、裁剪尺度、配准误差不统一
- token 数可能随图像尺寸动态变化
- 强行做 token MSE，优化约束会比监督信息更假
- 这会把工程重点从“得到可用 adapter”拉偏到“处理伪精确对齐”

因此 V1 loss 固定为：

```text
L_total = 1.0 * L_contrast + 1.0 * L_global_cosine
```

这里只做：

- 双向对比学习
- global cosine distillation

### 4.2 不在 V1 做真实无人机闭环控制

V1 只解决：

- `uav_image -> aligned_uav_tokens`

V1 不解决：

- ROS/PX4/MAVLink 接入
- 飞控命令安全约束
- 真机实时控制频率
- 丢帧/通信中断/姿态飘移兜底

---

## 5. 总体实现路线

分三阶段做。

### 阶段 A：Pair Alignment 预训练

输入：

- `(uav_image, sat_image)` 配对图

目标：

- 训练一个 `uav_adapter`
- 让 UAV 图经 teacher visual tower 编码后，再经过 adapter，落到 teacher 使用的 satellite visual space

输出：

- `uav_adapter` 权重
- projection head 权重
- 训练 manifest
- 检索/对齐评估结果

### 阶段 B：接入 OverlapVLN 推理链路

目标：

- 在 OverlapVLN 现有 `embed_enhance` 位置加载 `uav_adapter`
- 让模型支持：
  - `use_uav_adapter=true`
  - `uav_adapter_path=...`

输出：

- 可在 OverlapVLN 推理中直接把 UAV 图编码成与 teacher policy 对齐的视觉 token

### 阶段 C：Policy-aware 微调

目标：

- 在 synthetic UAV-like 数据或后续真实 UAV VLN 数据上继续训练
- 优先只调：
  - `uav_adapter`
  - 可选 `pose_embed`
  - 必要时再解冻少量 visual tail / history components

---

## 6. 推荐的代码布局

## 6.1 新增 package

建议新增：

- `src/swiftvln/s2r/`

理由：

- 不再使用 `adaptation/uav_sat_align` 这种两层语义叠加命名
- `s2r` 更短，后续实现和调用都更干净
- 比单独叫 `adapter` 更清楚，不容易和模型内部 LoRA / MLP adapter / token adapter 混淆
- `s2r` 对应 sim-to-real，语义足够明确，也能覆盖后续真实无人机迁移目标
- 这是一个跨任务模块，不应直接塞进 `model/`
- 后续如果 StreamVLN / MonoVLN / 其他真机模块复用，也更干净
- 与 `src/swiftvln/common/embedding_enhancement/` 的关系也清晰：
  - `s2r/` 负责训练与评估
  - `embedding_enhancement/` 负责在线接入

## 6.2 计划中的文件清单

建议新增这些文件：

- `src/swiftvln/s2r/__init__.py`
- `src/swiftvln/s2r/arguments.py`
- `src/swiftvln/s2r/dataset.py`
- `src/swiftvln/s2r/model.py`
- `src/swiftvln/s2r/losses.py`
- `src/swiftvln/s2r/trainer.py`
- `src/swiftvln/s2r/eval.py`
- `src/swiftvln/s2r/scripts/build_manifest.py`
- `src/swiftvln/s2r/scripts/train_s2r.sh`

同时新增一个可接入 OverlapVLN 的模块：

- `src/swiftvln/common/embedding_enhancement/uav_adapter.py`

并更新这些现有文件：

- `src/swiftvln/common/embedding_enhancement/__init__.py`
- `src/swiftvln/common/embedding_enhancement/pipeline.py`
- `src/swiftvln/model/arguments.py`
- `src/swiftvln/model/trainer.py`
- `src/swiftvln/model/model.py`

---

## 7. 阶段 A：Pair Alignment 训练设计

## 7.1 输入与 teacher 选择

Stage A 不走 OverlapVLN template，不走 history，不走 instruction。

只做：

```text
uav_image -> teacher_visual -> uav_adapter -> uav_tokens
sat_image -> teacher_visual -> sat_tokens
```

这里的 `teacher_visual` 默认来自目标 OverlapVLN checkpoint 的 visual tower。

## 7.2 V1 模型结构

V1 固定为：

- `teacher_visual`: frozen
- `uav_adapter`: trainable
- `projection_head`: trainable

其中：

- `uav_adapter` 作用在 token 级别
- `projection_head` 只用于 contrastive

### 7.2.1 `uav_adapter` 的具体选型

V1 不做复杂多分支。

直接实现为：

- `2` 个 token transformer block
- pre-norm
- residual
- hidden dim 与 teacher token dim 一致

选择这个版本的原因：

- 比单纯 MLP adapter 更能处理视角域差异
- 比复制整套 vision encoder 轻很多
- 后续可以原样挂进 `embed_enhance` pipeline

如果 V1 训练不稳定，再退回：

- residual MLP adapter

但默认先不上 MLP-only。

## 7.3 Loss

V1 固定为：

```text
L_total = L_contrast + L_global_cosine
```

具体定义：

- `g_uav = normalize(mean_pool(proj(uav_tokens)))`
- `g_sat = normalize(mean_pool(proj(sat_tokens)))`
- `L_contrast = CE(sim(g_uav, g_sat)) + CE(sim(g_sat, g_uav))`
- `L_global_cosine = 1 - cosine(mean_pool(uav_tokens), mean_pool(sat_tokens))`

不做：

- token distill
- local alignment
- patch matching

## 7.4 Manifest 格式

统一生成一个 manifest jsonl：

```json
{
  "dataset": "denseuav",
  "pair_id": "denseuav:000000_H80",
  "split": "train",
  "uav_image": "/mnt/data3/jiangjiajun/dataset/SatDronePair/denseuav/drone/000000_H80.jpg",
  "sat_image": "/mnt/data3/jiangjiajun/dataset/SatDronePair/denseuav/satellite/000000_H80.jpg",
  "group_id": "denseuav:train",
  "lat": 30.32413772,
  "lon": 120.38756039,
  "height_m": 80.0,
  "north_up_rot": 269.0,
  "meta": {
    "source": "pairs.csv"
  }
}
```

统一输出目录建议：

- `runtime/s2r/manifests/manifest_v1.jsonl`

## 7.5 训练输出目录

建议单独放：

- `output/s2r/<EXP_NAME>/`

其中包含：

- `adapter.pt` 或 safetensors
- `projection_head.pt`
- `train_args.json`
- `manifest_snapshot.json`
- `metrics.json`
- `train.log`

## 7.6 评估指标

Stage A 只看这几类：

- `uav -> sat Recall@1/5/10`
- `sat -> uav Recall@1/5/10`
- paired cosine similarity
- per-source retrieval
- cross-source retrieval

必须分源统计：

- DenseUAV
- GTA
- SUES
- UAVVisLoc

原因：

- 四个数据源分布差异大
- 混合均值会掩盖真实问题

## 7.7 V1 通过标准

V1 不要求一上来就能驱动无人机。

V1 通过标准定义为：

- mixed retrieval 明显高于随机基线
- 四个数据源里至少三个源 Recall@1 明显可用
- paired cosine 稳定提升
- adapter 输出能无缝接入 OverlapVLN 的视觉 token 接口

---

## 8. 阶段 B：接入 OverlapVLN

## 8.1 接入位置

直接复用现有 enhancement pipeline。

当前接入点已经存在：

- [src/swiftvln/model/template.py](/mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln/model/template.py#L578)

因此 `uav_adapter` 应实现为一个新的 enhancement module，接口形态与 `pixel_embed` / `pose_embed` 一致：

```python
embed = uav_adapter(embed, H, W, **kwargs)
```

## 8.2 新增参数

在 OverlapVLN 增加这些开关：

- `use_uav_adapter: bool = False`
- `uav_adapter_path: str = ""`
- `uav_adapter_type: str = "transformer_v1"`
- `uav_adapter_apply_scope: str = "all_images"`

V1 默认：

- `use_uav_adapter=false`
- `uav_adapter_apply_scope=all_images`

原因：

- 一旦进入真实 UAV 推理或 UAV-like synthetic finetune，输入流本身就是 UAV 图
- 不需要复杂的 per-image source gating

## 8.3 加载方式

在 `model.py` / `trainer.py` 中：

- 若 `use_uav_adapter=true`
- 则在 `embed_enhance` pipeline 中注册 `uav_adapter`
- 若 `uav_adapter_path` 非空，则从外部 checkpoint 恢复

权重恢复逻辑参考现有：

- `embed_enhance.*`

不要新搞一套不兼容保存逻辑。

## 8.4 阶段 B 的目标

阶段 B 不是追求性能最优，而是完成：

- 权重可加载
- forward 正常
- 输出形状与 dtype 正常
- history processor / pixel_embed / pose_embed 不被破坏

---

## 9. 阶段 C：Policy-aware 微调

## 9.1 为什么还需要这一步

Stage A 只保证：

- 视觉特征空间对齐

但真实目标是：

- 对真实 UAV 图也能做导航决策

所以后续仍需要 policy-aware 训练。

## 9.2 V1 的 policy-aware 训练建议

顺序如下：

1. 从当前最强 OverlapVLN teacher checkpoint 初始化
2. 挂载 Stage A 训练好的 `uav_adapter`
3. 先冻结：
   - LLM
   - 大部分 vision tower
   - 大部分 history processor
4. 只训练：
   - `uav_adapter`
   - 可选 `pose_embed`
   - 必要时少量 visual tail

## 9.3 训练数据优先级

优先级如下：

1. Google Earth / synthetic UAV-like VLN 数据
2. 真实 UAV 轨迹数据
3. 若暂时没有真实轨迹，再考虑少量人工采样验证集

V1 不建议直接拿真实 pair 数据硬做 action supervision。

---

## 10. 面向真实无人机部署的附加要求

这部分不是 Stage A 代码范围，但必须提前纳入设计约束。

## 10.1 真机部署还差的模块

仓库当前没有成型的：

- ROS/ROS2 runtime
- PX4/MAVROS/MAVLink bridge
- 实时状态同步
- 安全接管逻辑
- 延迟预算与降频策略

所以“pair alignment 成功”不等于“真机可飞”。

## 10.2 与真机最相关的信号

从部署角度看，后续最有价值的额外输入不是 token distill，而是：

- `heading`
- `GPS / relative pose`
- `altitude`
- `camera stabilization state`

这也是为什么 Stage C 优先建议继续保留和利用现有 `pose_embed` 能力，而不是做 token distill。

## 10.3 真机部署前必须验证的事情

- 实时推理吞吐
- 单帧与滑窗模式下的延迟
- heading 抖动下的输出稳定性
- 低质量图像与模糊图的退化行为
- GNSS 漂移或 north-up 估计误差对决策的影响

---

## 11. 实施顺序

建议严格按下面顺序来。

### Step 1

实现 manifest 构建脚本：

- `build_manifest.py`

产出：

- 统一 jsonl
- 每源统计
- split 统计

### Step 2

实现 Stage A dataset / trainer / eval。

目标：

- 跑通 pair alignment
- 产出 adapter checkpoint

### Step 3

实现 `uav_adapter.py` 并接入 `embed_enhance` pipeline。

目标：

- 可以在 OverlapVLN 里 load adapter

### Step 4

做 OverlapVLN 接入 smoke test。

只验证：

- 权重加载
- forward 正常
- 不破坏现有评测链路

### Step 5

再考虑 synthetic UAV-like policy-aware finetune。

---

## 12. 本文档对应的最终实现决策

V1 的最终决策固定如下：

- **不做 token distill**
- **teacher 对齐到目标 OverlapVLN checkpoint 的 visual space**
- **Stage A 单独建训练分支，不复用现有导航 dataset/trainer**
- **新训练模块放 `src/swiftvln/s2r/`**
- **在线接入模块放 `src/swiftvln/common/embedding_enhancement/uav_adapter.py`**
- **Stage B 通过现有 `embed_enhance` pipeline 接入 OverlapVLN**
- **Stage C 再做 policy-aware 微调**

如果后续实现时出现资源瓶颈，优先裁剪顺序如下：

1. 先减少数据源，保留 `DenseUAV + GTA`
2. 再降低 adapter 深度
3. 再减少 projection head 输出维度

不要优先裁掉：

- global cosine distill
- 分源评估
- OverlapVLN 接口兼容性验证
