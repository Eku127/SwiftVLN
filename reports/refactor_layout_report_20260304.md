# SwiftVLN 重构实施报告（2026-03-04）

## 1. 本次实施范围

按既定计划实现以下内容：
- 移除 `examples/vln` 路径，重排为包结构 `src/swiftvln`。
- 核心三模型（`streamvln` / `compressvln` / `overlapvln`）完成入口与导入重构。
- 新增独立仓库打包能力（`pyproject.toml`）与统一 CLI（`swiftvln`）。
- 保留并迁移现有 shell 编排脚本到 `src/swiftvln/scripts`。

## 2. 目录结构变化

旧结构（已移除）：
- `examples/vln/*`

新结构：
- `src/swiftvln/common`
- `src/swiftvln/models/{streamvln,compressvln,overlapvln,monovln,uninavid}`
- `src/swiftvln/configs`
- `src/swiftvln/scripts`
- `src/swiftvln/runners`
- `src/swiftvln/cli.py`
- `pyproject.toml`

## 3. 核心改造点

### 3.1 包化与 CLI
- 新增 `pyproject.toml`（editable install 可用）。
- 新增控制台命令：`swiftvln`。
- CLI 子命令：
  - `swiftvln train --model <model> -- <args...>`
  - `swiftvln eval --model <model> -- <args...>`
  - `swiftvln queue {train|eval}`

### 3.2 核心三模型导入重构
对 `streamvln/compressvln/overlapvln`：
- 移除训练/评估入口中的 `sys.path.insert` 与 `_msswift_root/_vln_dir` 注入。
- 移除 `try/except ImportError` 的本地 fallback 导入。
- 统一为绝对包导入：`swiftvln.*`。

### 3.3 通用模块重构
- `common/base_eval.py`
  - 移除运行期 `sys.path` 注入。
  - 配置默认路径从 `config/*.yaml` 调整为 `configs/*.yaml`。
  - 配置解析逻辑改为以 package root / repo root 双路径解析。
- `common/base_evaluator.py`
  - 简化 habitat_extensions 注册导入逻辑，移除 fallback 路径注入。

### 3.4 脚本路径与运行环境
- 所有脚本路径切换到 `src/swiftvln/...`。
- 含 `SWIFTVLN_ROOT` 的 shell 脚本统一注入：
  - `export PYTHONPATH="${SWIFTVLN_ROOT}/src:${PYTHONPATH:-}"`
- 修复 `train_queue.sh` 内临时脚本替换路径，指向 `src/swiftvln/models/${model}`。

## 4. 验证结果

### 4.1 静态验证
- `bash -n` 全量 shell 脚本通过。
- `python -m compileall`（core3 + common + cli + runners）通过。

### 4.2 包与 CLI 验证
- `pip install -e .` 成功。
- `python -c "import swiftvln"` 成功。
- `swiftvln --help`、`swiftvln train --help`、`swiftvln eval --help`、`swiftvln queue --help` 成功。

### 4.3 运行验证（真实 smoke）
- 在 `swift-vln-eval` 环境下，以新入口执行并成功完成：
  - `python -m swiftvln.models.streamvln.eval ... --habitat_config_path configs/vln_r2r_smoke.yaml --max_episodes 1`
- 结果输出：
  - `results/refactor_smoke/streamvln_habitat/evaluation_summary.json`
  - 日志：`logs/refactor_smoke/eval_streamvln_habitat.log`

## 5. 当前已知剩余项

本次按“核心3模型优先”策略执行，以下区域仍保留旧式路径注入（未纳入本次核心改造）：
- `models/monovln/*`
- `models/uninavid/*`
- `tools/feature_test/*`
- `scripts/vit_feat_precompute/precompute_features.py`

这些不影响本次核心三模型主线（train/eval/queue）的新包路径运行。

## 6. 结论

- SwiftVLN 已完成从 `examples/vln` 到 `src/swiftvln` 的核心独立化重构。
- 核心三模型架构主线已切换到新包路径并通过静态+运行验证。
- 仓库已具备独立包安装与统一 CLI 入口能力。
