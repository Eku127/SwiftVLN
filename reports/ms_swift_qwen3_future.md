# ms-swift 4.x 适配与 Qwen3/Qwen3.5 可行性结论

日期：2026-04-30

## 当前适配基线

- PyPI 最新稳定版核对为 `ms-swift 4.1.1`（2026-04-13 发布）。
- 本次按用户指定的本地源码 `/mnt/data1/home/jiangjiajun/workspace/ms-swift-lateset` 适73配；该仓库当前为 `main`，commit `ad7d5c515 [docs] fix docs (#9244)`，版本号 `4.2.0.dev0`。
- 原有 `/mnt/data1/home/jiangjiajun/workspace/ms-swift` 是 `3.12.0.dev0`，主要旧入口 `swift.llm.*` 在 4.x 已拆分到 `swift.arguments`、`swift.dataset`、`swift.model`、`swift.template`、`swift.pipelines.train.sft`。

参考来源：
- PyPI: https://pypi.org/project/ms-swift/
- GitHub: https://github.com/modelscope/ms-swift
- Qwen3.5 best practices: https://swift.readthedocs.io/en/v4.1/BestPractices/Qwen3_5-Best-Practice.html

## 本次结构调整

- SwiftVLN 内部直接使用 ms-swift 4.x API，不再保留 ms-swift 3.x 兼容层。
- OverlapVLN 使用 `ModelLoader` 子类注册；旧 3.x `get_function` 注册路径已移除。
- OverlapVLN 特殊 token 注入迁移到 4.x loader 的 `new_special_tokens` 流程，继续包含：
  - `<history_image>`
  - `<history_memory>`
  - `<current_image>`
- 训练和评测入口脚本名称及参数保持不变，仅默认激活环境切到：
  - train: `swift-vln-train-update`
  - eval: `swift-vln-eval-update`
  可继续通过 `SWIFTVLN_TRAIN_CONDA_ENV` / `SWIFTVLN_EVAL_CONDA_ENV` 覆盖。

## Qwen3 / Qwen3.5 可行性

结论：有可能适配，但不是简单替换 `model_type` 和 `model_path`。OverlapVLN 当前强绑定 Qwen2.5-VL 的模型类、processor 行为、vision token 展开和 template 后处理；Qwen3/Qwen3.5 需要做一层“模型族抽象”。

需要的结构调整：

- 模型层：新增 Qwen3-VL / Qwen3.5 对应的 OverlapVLN subclass 和 config，不能复用 `Qwen2_5_VLForConditionalGeneration`。
- Loader 层：按 ms-swift 4.x 的 `Qwen3VLLoader` / `Qwen3_5Loader` 派生新的 OverlapVLN loader，并复用特殊 token 注入与 embed_enhance 附加逻辑。
- Template 层：从当前 `Qwen2_5VLTemplate` 继承改为按模型族继承 `Qwen3VLTemplate` 或 `Qwen3_5Template`；重点复核 `_encode()`、`_post_encode()`、image/video token id、grid/thw 与 RoPE 处理。
- 训练脚本层：保留外部脚本风格，但需要把 base model 路径、`model_type`、template/model family 配置参数化，避免 Qwen2.5 写死。
- 依赖层：Qwen3-VL 在最新 ms-swift 中要求更高版本 `transformers` 与 `qwen_vl_utils>=0.0.14`；Qwen3.5 文档建议 `transformers==5.2.*`、`qwen_vl_utils>=0.0.14`，可能需要独立 conda 环境，避免破坏当前 Qwen2.5-VL 训练栈。

建议：先把当前 Qwen2.5-VL 适配稳定跑通完整 train/eval smoke，再单独开 `overlapvln_qwen3_vl` 分支型注册；Qwen3.5 优先做 `load_model=False` processor/template 验证，再做小样本训练，因为 Transformers 5.x 依赖面更大。
