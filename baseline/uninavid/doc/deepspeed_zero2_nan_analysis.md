# DeepSpeed ZeRO-2 Loss 崩溃根因分析

> 日期：2026-03-12  
> 环境：server 98，8× H100 80GB，transformers 4.34.1，deepspeed 0.18.7，BF16，gradient_checkpointing=True

## 问题现象

在 SatNav 全量数据（ver_260306）上用 `zero2.json` 训练 Uni-NaVid，step 19 起 `loss_raw` 持续打印 `0.0`，且在 73 和 98 服务器上均可复现。

```
step 17: {'loss': 0.4341, 'loss_raw': 0.43408203}
step 18: {'loss': 0.4143, 'loss_raw': 0.41430664}
step 19: {'loss': 0.0,    'loss_raw': 0.0}   ← 突然崩溃
step 20: {'loss': 0.0,    'loss_raw': 0.0}
...（此后持续为 0）
```

## 根因一：`loss_raw=0.0` 不是真正的 0，是被过滤的 NaN

transformers 4.34.1 的 `Trainer.train()` 默认 `logging_nan_inf_filter=True`：

```python
# trainer.py L1894-1902
if (
    args.logging_nan_inf_filter
    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
):
    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
else:
    tr_loss += tr_loss_step
```

当 `logging_steps=1` 时，每步 log 后 `tr_loss` 被清零。所以 NaN 步骤的替换值是 `0 / 1 = 0`，看起来就像 loss 真的是 0。

**`loss_raw` 字段同样受影响**，因为它从 `tr_loss_scalar` 派生，而 `tr_loss_scalar` 是在过滤之后才被计算的。

## 根因二：`overlap_comm=True` 与 `gradient_checkpointing` 的交互

### zero2.json 的问题配置

```json
"zero_optimization": {
    "stage": 2,
    "overlap_comm": true,       ← 根因
    "contiguous_gradients": true,  ← 加剧因素
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto"
}
```

### 崩溃机制

1. ZeRO-2 的 `overlap_comm=True` 会在每个参数的梯度就绪时，立即注册 hook 触发 reduce-scatter
2. `gradient_checkpointing=True` 的 backward 需要对每个 checkpoint 段**重新执行 forward**（recompute）来重算激活值
3. 在**重计算 forward** 阶段（处于 `torch.enable_grad()` 上下文中），autograd 再次遍历部分参数节点
4. 此时一些参数的梯度 hook 被**提前触发**，用不完整的梯度做了 reduce-scatter
5. reduce-scatter 完成后，该参数的梯度 buffer 被清空（每个 rank 只保留 1/8 的 shard）
6. 真正的梯度随后写入已清空的 buffer，导致梯度被覆盖或多次 reduce
7. 经过约 18 步累积，模型参数出现 NaN/Inf
8. forward 输出 NaN loss → 被 `logging_nan_inf_filter` 替换为 0

`contiguous_gradients=True` 将所有梯度集中到单一连续 buffer，使上述问题更容易触发（更大的 buffer 意味着更多参数的 hook 共享同一段内存）。

## 验证实验

### 对照配置（zero2_safe.json）

```json
"zero_optimization": {
    "stage": 2,
    "overlap_comm": false,        ← 关闭
    "contiguous_gradients": false, ← 关闭
    "reduce_bucket_size": "auto"
}
```

| 配置 | step 19 loss | 全程 40 步 | 结论 |
|---|---|---|---|
| `zero2.json`（原始）| `0.0`（实为 NaN）| 崩溃 | ❌ |
| `zero2_safe.json` | `0.162`（正常）| 全部正常 | ✅ |
| `zero1.json` | `0.159`（正常）| 全部正常 | ✅ |

### debug 验证

用 `UNINAVID_DEBUG_LOSS=1` + `MAX_STEPS=25` 跑 ZeRO-2，发现全程无 NaN（因为 `MAX_STEPS=25` 压缩了 LR 调度，warmup 基本在 step 1 完成，回避了 warmup 阶段的梯度放大问题）。

## 推荐方案

**正式训练使用 `zero1.json`**，理由：
- 在 98 服务器 8× H100 80GB、`TRAIN_BSZ=24` 下不 OOM，ZeRO-2 的显存优势不必要
- ZeRO-1 已充分验证（73 服务器 120 steps + 98 服务器多次验证）
- `zero2_safe.json` 关掉 `overlap_comm` 后性能优势几乎消失，不如 ZeRO-1 稳定

若未来遇到 ZeRO-1 OOM（更大模型或更大 bsz），可切换到 `zero2_safe.json`，但需要先跑完整 smoke test。

## 相关文件

| 文件 | 说明 |
|---|---|
| `baseline/uninavid/configs/zero1.json` | 当前默认，稳定 |
| `baseline/uninavid/configs/zero2.json` | 原始问题版，保留供参考 |
| `baseline/uninavid/configs/zero2_safe.json` | 修复版，已验证 40 步正常 |
