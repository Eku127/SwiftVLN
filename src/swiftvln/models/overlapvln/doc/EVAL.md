# OverlapVLN 评估流程详解

本文档详细描述 OverlapVLN Evaluator 的实现细节和工作流程。

---

## 1. 评估器架构

```python
class OverlapVLNEvaluator(BaseVLNEvaluator):
    # 核心缓存
    history_cache: List[Tensor]     # 压缩后的全局历史帧特征
    overlap_context: OverlapContext  # 上一窗口的最后4轮
    window_turns: List[TurnContext]  # 当前窗口已完成的轮次
    
    # 状态追踪
    window_start_step: int   # 当前窗口起始步数
    current_window_idx: int  # 当前窗口编号
```

---

## 2. 核心流程

### 2.1 Episode 评估主循环

```python
def eval_episode(env_wrapper, episode):
    # 初始化
    _reset_caches()
    rgb_list = []
    action_seq = []
    step_id = 0
    
    while not done and step_id < max_steps:
        # 1. 收集观察
        current_img = env.get_rgb()
        rgb_list.append(current_img)
        
        # 2. 如果需要预测新动作
        if len(action_seq) == 0:
            # 2a. 检查窗口滑动
            if should_slide_window(step_id):
                _slide_window(rgb_list, new_window_start)
            
            # 2b. 构建 prompt embedding
            inputs_embeds, seq_len, current_vit = _build_complete_prompt_embeds(
                instruction, current_img, conjunction
            )
            
            # 2c. 生成响应
            response = model.generate(inputs_embeds)
            action_seq = parse_actions(response)
            
            # 2d. 保存轮次
            _save_turn_to_window(conjunction, response, current_vit)
        
        # 3. 执行动作
        action = action_seq.pop(0)
        env.step(action)
        step_id += 1
    
    return env.get_metrics()
```

### 2.2 窗口滑动条件

```python
def should_slide_window(step_id):
    return (
        step_id > 0 and                    # 不是第一步
        step_id % stride == 0 and          # 到达滑动点
        step_id >= num_frames              # 已完成至少一个完整窗口
    )
```

示例（num_frames=32, stride=16）：
```
step_id=0:   不滑动（第一步）
step_id=16:  不滑动（未完成第一个窗口）
step_id=32:  滑动！（完成第一个窗口）
step_id=48:  滑动！
step_id=64:  滑动！
```

---

## 3. 窗口滑动详解

### 3.1 _slide_window 流程

```python
def _slide_window(rgb_list, new_window_start):
    # 1. 保存 overlap context
    _prepare_overlap_context()
    
    # 2. 重新计算历史缓存
    _compute_history_cache(rgb_list, new_window_start)
    
    # 3. 重置窗口状态
    window_turns = []
    window_start_step = new_window_start
    current_window_idx += 1
```

### 3.2 _prepare_overlap_context

```python
def _prepare_overlap_context():
    # 取当前窗口最后 overlap_turns 个 turn
    overlap_turns_list = window_turns[-overlap_turns:]  # 最后4轮
    
    # 构建拼接的 input_ids
    ids_parts = []
    image_embeds = []
    
    for turn in overlap_turns_list:
        ids_parts.append(turn.user_input_ids)  # user turn tokens
        ids_parts.append(_build_assistant_turn_ids(turn.assistant_response))
        image_embeds.append(turn.image_embed)
    
    overlap_context = OverlapContext(
        input_ids=torch.cat(ids_parts, dim=1),
        image_embeds=image_embeds,
    )
```

### 3.3 _compute_history_cache

```python
def _compute_history_cache(rgb_list, window_start):
    if window_start <= 0:
        history_cache = []
        return
    
    # 从 [0, window_start) 采样 num_history 帧
    history_indices = sample_history_indices(window_start, num_history)
    history_images = [rgb_list[i] for i in history_indices]
    
    # 批量编码
    features_list, grid_thw_list = _encode_batch_frames(history_images)
    
    # 压缩每张
    history_cache = []
    for features, grid_thw in zip(features_list, grid_thw_list):
        compressed = _compress_features(features, grid_thw)
        history_cache.append(compressed)
```

---

## 4. Embedding 构建详解

### 4.1 _build_complete_prompt_embeds

```
输入:
  - instruction: 导航指令
  - current_image: 当前帧 PIL.Image
  - conjunction: 随机连接词

输出:
  - inputs_embeds: [1, seq_len, hidden_size]
  - seq_len: 序列长度
  - current_vit_features: [num_tokens, hidden_size]
```

### 4.2 构建步骤

```
Step 1: 编码当前帧
        current_vit_features = model.visual(current_image)
        current_token_count = current_vit_features.shape[0]  # 256

Step 2: 构建 system prompt
        history_token_counts = [h.shape[0] for h in history_cache]  # [64, 64, ...]
        system_ids = _build_system_prompt_ids(instruction, history_token_counts)
        system_embeds = embed_tokens(system_ids)

Step 3: 替换历史图像 token
        history_positions = find_positions(system_ids, history_image_token_id)
        history_embeds = cat(history_cache)
        system_embeds[0, history_positions] = history_embeds

Step 4: 添加 overlap_context (如果有)
        overlap_embeds = embed_tokens(overlap_context.input_ids)
        overlap_positions = find_positions(overlap_context.input_ids, current_image_token_id)
        overlap_img_embeds = cat(overlap_context.image_embeds)
        overlap_embeds[0, overlap_positions] = overlap_img_embeds

Step 5: 添加当前窗口已完成的 turns
        for turn in window_turns:
            turn_user_embeds = embed_tokens(turn.user_input_ids)
            turn_positions = find_positions(turn.user_input_ids, current_image_token_id)
            turn_user_embeds[0, turn_positions] = turn.image_embed
            # ... 加上 assistant response

Step 6: 添加新的 user turn (带 generation prompt)
        new_user_ids = _build_user_turn_ids(conjunction, current_token_count, add_generation_prompt=True)
        new_user_embeds = embed_tokens(new_user_ids)
        cur_positions = find_positions(new_user_ids, current_image_token_id)
        new_user_embeds[0, cur_positions] = current_vit_features

Step 7: 拼接所有部分
        inputs_embeds = cat([system_embeds, overlap_embeds, window_turn_embeds, new_user_embeds])
```

---

## 5. Token ID 构建

### 5.1 _build_system_prompt_ids

```python
def _build_system_prompt_ids(instruction, history_token_counts):
    """
    构建包含历史帧占位符的 system prompt token IDs
    
    Args:
        instruction: 导航指令
        history_token_counts: 每张历史帧的 token 数量列表，如 [64, 64, 64, ...]
    
    Returns:
        [1, seq_len] 的 token IDs
    """
    system_prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    
    if history_token_counts:
        # 为每张历史帧生成正确数量的占位符 token
        history_parts = []
        for token_count in history_token_counts:
            tokens = "<history_image>" * token_count  # 64 个 token
            history_parts.append(f'<|vision_start|>{tokens}<|vision_end|>')
        
        history_tokens = ' '.join(history_parts)
        system_prompt += f" These are your historical observations: {history_tokens}."
    
    return tokenize(system_prompt)
```

### 5.2 _build_user_turn_ids

```python
def _build_user_turn_ids(conjunction, current_token_count, add_generation_prompt=False):
    """
    构建 user turn 的 token IDs
    
    Args:
        conjunction: 连接词，如 "you can see"
        current_token_count: 当前帧的 token 数量，如 256
        add_generation_prompt: 是否添加生成提示（只有最后一个 turn 需要）
    
    Returns:
        [1, seq_len] 的 token IDs
    """
    current_tokens = "<current_image>" * current_token_count  # 256 个 token
    content = f"{conjunction}<|vision_start|>{current_tokens}<|vision_end|>."
    
    messages = [{'role': 'user', 'content': content}]
    return apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
```

**关键**: `add_generation_prompt` 参数
- 已完成的 turns: `False` (user + assistant 完整对话)
- 最后的 new turn: `True` (让模型生成 assistant response)

---

## 6. 图像 Token 替换机制

### 6.1 索引赋值 vs masked_scatter

```python
# ❌ 错误方式: masked_scatter 只会使用 source 的前 N 个元素
# mask 中有 2048 个 True (一个位置的 hidden_size)
# source 有 256*2048 个元素
# 只有 source[0] 被使用
mask = (input_ids == token_id).unsqueeze(-1).expand_as(embeds)
embeds = embeds.masked_scatter(mask, source)

# ✅ 正确方式: 索引赋值
# positions 有 256 个位置
# source 有 256 行
# 一一对应
positions = (input_ids[0] == token_id).nonzero(as_tuple=True)[0]
embeds[0, positions[:256]] = source
```

### 6.2 多图像处理

```python
# overlap_context 有 4 张图像，每张 256 tokens
# overlap_context.image_embeds = [img1, img2, img3, img4]
# 每个 imgi: [256, 2048]

# 拼接: [1024, 2048]
all_embeds = torch.cat(overlap_context.image_embeds, dim=0)

# positions 有 1024 个位置
positions = find_positions(input_ids, current_image_token_id)

# 按顺序填入
embeds[0, positions[:1024]] = all_embeds
```

---

## 7. 性能优化

### 7.1 VIT 特征缓存

```
首次编码:  image → image_processor → pixel_values → model.visual() → features
后续使用:  features (已缓存)

节省:
- history_cache: 避免重复编码历史帧
- window_turns[].image_embed: 避免重复编码已见过的当前帧
- overlap_context.image_embeds: 窗口间复用
```

### 7.2 Token ID 缓存

```
保存 input_ids 而非 embedding:
- TurnContext.user_input_ids: int tensor [1, seq_len]
- OverlapContext.input_ids: int tensor [1, seq_len]

优势:
- 内存占用小 (int vs float)
- 使用时调用 embed_tokens() 很快
```

---

## 8. 调试信息

### 8.1 启用 verbose 模式

```bash
python eval.py ... --verbose
```

输出示例：
```
Step 0: window=0, turns=1, history=0, output=↑ ↑ ↑ ↑
Step 4: window=0, turns=2, history=0, output=↑ ↑ ← ←
...
Step 32: window=1, turns=1, history=8, output=↑ ↑ ↑ →
```

### 8.2 启用 timing 模式

```bash
python eval.py ... --debug_timing
```

输出示例：
```
============================================================
Episode ep_001 (Scene: scene_001) Timing Statistics:
============================================================
Total episode time: 15.234s
Total steps: 48
Windows used: 2
History cache size: 8

Breakdown by component:
  model_generate      :    8.521s (55.9%)
  build_embeds        :    3.124s (20.5%)
  window_slide        :    1.234s ( 8.1%)
  ...
============================================================
```

---

## 9. 常见问题

### Q1: 为什么模型输出乱码？

**可能原因**: Chat template 格式错误

检查点:
- `_build_user_turn_ids` 的 `add_generation_prompt` 参数
- 只有最后的 user turn 应该设置 `True`

### Q2: 为什么图像特征没有正确填入？

**可能原因**: 图像 token 数量不匹配

检查点:
- `positions` 的数量应该等于 `image_embed.shape[0]`
- 使用索引赋值而非 masked_scatter

### Q3: 窗口滑动后性能下降？

**可能原因**: History cache 或 overlap_context 未正确传递

检查点:
- `_prepare_overlap_context()` 是否保存了正确的 turns
- `_compute_history_cache()` 是否从正确的范围采样
