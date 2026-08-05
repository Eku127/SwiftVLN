# SwiftVLN 代码精简审计报告

> 初始审计日期：2026-08-05<br>
> 审计范围：`src/swiftvln/`<br>
> 初始审计约束：只审计、分类和规划，不删除、不重构任何源码<br>
> 实施更新：2026-08-05 已完成 A01–A07 及第 5 节确认范围内的结构性精简；本文档同步记录实现、验证、清理与原子提交<br>
> 当前状态：重构代码、契约测试、Qwen2.5/Qwen3 × SatNav/Habitat 实机矩阵均已完成，专用测试产物已清理<br>
> 结论性质：以静态引用、当前入口、仓库模型库和默认事实文档为依据；涉及外部调用方的项目均列为“需确认”，不直接判死

## 1. 结论摘要

初始审计发现，`src/swiftvln` 的主要问题并不是某一个大文件，而是四类历史设计叠加：

1. **单模型仓库仍保留多模型框架抽象**：CLI、registry、runner、训练/评测 Base 类都在为不存在的第二个主线模型提供间接层。
2. **同一份实验配置存在多套事实源**：训练参数、评测参数、训练 shell、评测 shell、模型名解析、部署解析和结果汇总各自维护规则，已经发生漂移。
3. **一次性工具和调试功能长期留在主包**：历史 SatNav split 脚本、R2R/UniNaVid 特征预计算、旧分析工具、landmark/map debug 代码占用了较多主流程体积。
4. **推理、数据处理和小型算法存在重复实现**：评测与部署维护两套窗口推理，多个 SatNav 脚本重复 I/O，GTC/SegmentGTC 重复 soft k-means 步骤。

本轮按以下顺序完成精简：

- 先补契约测试和模型名 golden cases；
- 再清理确定无行为影响的注释、无用变量、废弃 API 和 no-op 参数；
- 然后归档一次性工具；
- 再统一配置与模型名解析；
- 最后处理评测/部署推理合并，以及 Base 类、CLI、队列交互层等结构性问题。

没有为了行数删除以下能力：GTC、SegmentGTC、map memory、initial prompt、pose、overlap/no-memory、Qwen2.5-VL、Qwen3-VL、SatNav/Habitat。当前模型库保存的这些变体仍可由统一实验 schema 解析；唯一主动收缩的是已由维护者明确不要的 pose 与 UAV 组合模式。

实施结论：初始审计的 141 个源码/脚本/配置/文档文件、33,272 行，现为 114 个文件、26,269 行，净减少 27 个文件和 7,003 行（约 21%）。deployment、多模型 registry/runners、历史数据/分析工具、单实现 Base 层和交互式队列向导已经移除；配置命名、评测推理、环境循环、结果记录和环境能力边界已结构化。HF wrapper/config 类名、模型类型、稳定训练/评测 shell 路径、S2R 主线及 SatNav/Habitat 能力保持不变。

## 2. 当前规模与健康度

### 2.1 代码规模（初始审计基线）

统计对象为 `src/swiftvln` 下的 Python、shell、YAML 和 Markdown 文件。

| 区域 | 文件数 | 行数 | 初步判断 |
|---|---:|---:|---|
| 根入口 | 3 | 83 | CLI 存在历史多模型抽象 |
| `common/` | 32 | 4,622 | 真正公用组件与单实现 Base 类混杂 |
| `configs/` | 5 | 303 | 应保留，但应减少与 shell/argparse 的重复默认值 |
| `deployment/` | 8 | 1,518 | 与评测重复，且当前默认解析不可用 |
| `habitat_extensions/` | 3 | 1,036 | 大量注释掉的历史实现和未使用地图工具 |
| `model/` | 15 | 7,665 | 主线核心；评测器过大并混入调试分析 |
| `runners/` | 5 | 125 | 只服务单一 `swiftvln` 模型的动态分发层 |
| `s2r/` | 35 | 6,949 | 当前 Stage-A 主线，应保留，仅做小范围去重 |
| `scripts/` | 35 | 10,971 | 当前编排脚本、一次性历史工具、分析脚本混杂 |
| **合计** | **141** | **33,272** | 108 Python、20 shell、6 YAML、7 Markdown |

Phase 1 A01–A07 完成后，文件数不变，源码总行数为 32,692；七个源码提交合计净减少 580 行。完成后续结构性精简的当前规模如下：

| 区域 | 当前文件数 | 当前行数 | 相对初始状态 |
|---|---:|---:|---|
| 根入口 | 4 | 754 | 新增统一 `experiment.py`，CLI 改为单模型直连 |
| `common/` | 28 | 3,706 | 移除单实现训练/评测 Base，保留组合组件 |
| `configs/` | 5 | 303 | 路径和能力不变 |
| `deployment/` | 0 | 0 | 按维护者决定整体删除 |
| `habitat_extensions/` | 2 | 59 | 只保留当前 YAML 使用的两个 measure |
| `model/` | 17 | 8,061 | evaluator 拆为 inference/runner/diagnostics 等职责 |
| `runners/` | 0 | 0 | 删除唯一模型的动态分发层 |
| `s2r/` | 35 | 6,949 | Stage-A 主线保持不变 |
| `scripts/` | 23 | 6,437 | 删除历史工具，收口 train/eval queue 协议 |
| **合计** | **114** | **26,269** | **较初始净减少 27 个文件、7,003 行** |

### 2.2 静态检查（初始审计基线）

- `python3 -m compileall -q src/swiftvln`：通过。
- 所有主线 shell 执行 `bash -n`：通过。
- `ruff check src/swiftvln --statistics`：87 个问题：
  - 35 个未使用 import；
  - 27 个无占位符 f-string；
  - 17 个非文件顶部 import；
  - 5 个未使用局部变量；
  - 2 个单行复合语句；
  - 1 个 bare `except`。
- 60 个问题可由 Ruff 常规安全修复，另有 4 个需要 unsafe fix；不能盲目全修，评测入口中的部分 E402 是为了在 Habitat import 前设置 NVIDIA EGL。
- 当前仓库只发现一个测试脚本：`src/swiftvln/model/script/test/test_uav_adapter_strategy.py`。缺少覆盖训练/评测核心契约的测试，是精简工作的首要风险。

最终实施后：

- Ruff 问题由 87 个降至 0；需要在路径/EGL 初始化后导入的少数位置使用显式 `# noqa: E402`，没有移动其运行时顺序。
- `python -m compileall -q src/swiftvln tests`、`src/swiftvln` 下全部 shell 的 `bash -n` 均通过。
- 新增 15 个契约测试文件，`python -m unittest discover -s tests -v` 共 57/57 通过。
- UAV adapter 独立 smoke、CLI/数据/S2R 入口 smoke 通过。
- Qwen2.5/Qwen3 × SatNav/Habitat 的训练和评测实机矩阵通过；完整结果见第 12 节。

## 3. 当前必须保留的边界

### 3.1 能力边界

当前模型库共保留 22 个 HF 模型，其中 `output/model_zoo/swiftvln/HF_model/` 有 11 个 SwiftVLN 变体，明确覆盖：

- per-frame baseline；
- GTC、SegmentGTC；
- map memory；
- no-memory；
- random/log sampling；
- initial prompt；
- pose Film；
- overlap 0/4/16；
- Qwen2.5-VL 与 Qwen3-VL（后者也存在于 backbone 模型库）。

因此，某项功能“不是默认值”不能作为删除理由。至少应先保证模型库中所有短模型名仍能解析、加载和评测。

### 3.2 当前规范入口

仓库当前事实文档和脚本使用的是：

- 训练：`src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh`，最终直接调用 `src/swiftvln/model/trainer.py`；
- 训练队列：`src/swiftvln/scripts/train/train_queue.sh`；
- 评测：`src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh`，最终调用 `python -m swiftvln.model.eval`；
- 按名称评测：`src/swiftvln/scripts/eval/eval_by_name.sh`；
- S2R 数据：`swiftvln s2r-data ...`；
- 评测结果：`all_results.jsonl`。

CLI 中的 `swiftvln train`、`swiftvln eval`、`swiftvln queue` 没有出现在当前规范工作流中。它们是否属于对外 API，需要由维护者确认。

## 4. 精简候选总表

分类定义：

- **A：可直接清理**——有明确静态证据，行为风险低，但仍建议随测试提交；
- **B：迁移或合并后清理**——能力仍需保留，只应删除重复实现或主流程中的附属代码；
- **C：需维护者决策**——可能有仓库外调用方，或涉及产品范围；
- **D：必须保留**——当前模型库、主线流程或技能明确依赖。

| ID | 候选 | 分类 | 证据与影响 | 建议 |
|---|---|---|---|---|
| A01 ✅ 已完成 | `habitat_extensions/measures.py` 中大段注释实现 | A | 630 行文件中 370 行为注释；约 221–630 行是完整注释掉的 WaypointReward/NDTW/SDTW/TopDownMap 实现 | 已删除注释块及仅服务这些注释的 import；保留 6 个活跃 measure。提交：`d352994`；验证：扩展模块注册和 Habitat 实流 smoke |
| A02 ✅ 已完成 | 模型 KV-cache 管理接口 | A/B | `SwiftVLNStreamingMixin` 的 `get_cache`、`update_cache`、`get_step_count` 无调用点；当前评测每步重建完整 prompt embeds | 已保留 HF wrapper/config 类名，删除无消费者的缓存状态 API 及 runner/deployment 中的 reset 调用。提交：`b4619f4`；验证：wrapper 契约、模型加载和 SatNav 评测 |
| A03 ✅ 已完成 | 已标记废弃和 no-op API | A | `get_compressed_length()` 标为 DEPRECATED 且无引用；`reconstruct_pose_from_actions(..., norm_scale)` 明确忽略参数；`run_all --episodes-only` 不改变流程 | 已删除废弃方法、无效参数和无效 flag。提交：`3e8b2a7`；验证：pose/compressor 契约及 `run_all --help` |
| A04 ✅ 已完成 | 明确无用局部状态 | A | Ruff 报告 5 个未使用变量；如 compressor 的 `dtype`、`row_cells`、`num_centers`，evaluator 的 `episode_exception` | 已删除 5 个未使用局部变量，并保持原异常处理行为。提交：`3d5079d`；验证：Ruff F841 清零、compileall 和集成 smoke |
| A05 ✅ 已完成 | GTC 与 SegmentGTC 的 `_soft_kmeans_step` | B | 两个 17 行方法 AST 完全一致 | 已提取共享私有函数，保留两种处理器及公开行为。提交：`f808ca5`；验证：固定输入数值等价及两种处理器 shape 契约 |
| A06 ✅ 已完成 | Qwen loader 的重复初始化 | B | Qwen2.5/Qwen3 loader 的两个 `__init__` 为相同实现 | 已提取共享 loader 参数 helper，保留两套注册类和 Qwen3 patch。提交：`5bd5707`；验证：参数契约和 Qwen3 2B 实际加载 |
| A07 ✅ 已完成 | 过宽异常吞噬 | B | `model.py` 注册处捕获所有 `Exception`；`model/eval.py` 的注册 import 捕获 `ImportError` 后直接 pass | 已改为 Transformers `exist_ok=True` 幂等注册，并让模块导入错误显式暴露。提交：`7d3ed13`；验证：重复 reload、扩展注册及 train/eval smoke |
| B01 ✅ 已完成 | 统一实验配置和模型名 schema | B | 参数/默认值曾分散在 arguments、eval argparse、训练/评测 shell、`eval_by_name.sh` 和 results collector | 新增 `SwiftVLNExperimentSpec`、统一 parse/build/validation；训练 shell、评测 shell 和结果汇总共用。提交：`332cff6` |
| B02 ✅ 已完成 | 评测与部署的推理核心 | B/C | 两边曾重复 frame encode、history cache、prompt 构造和滑窗逻辑 | deployment 按决定删除；保留逻辑提取为 `SwiftVLNInferenceSession`。提交：`c1cb769`、`22ee635` |
| B03 ✅ 已完成 | evaluator 内 landmark 调试分析 | B | landmark failure 分析曾混在 episode 主循环 | 移入独立 `model/diagnostics.py`，核心 evaluator 不再生成调试报告。提交：`22ee635` |
| B04 ✅ 已完成 | evaluator/map_memory 的 map debug | B | map prompt/tensor dump 曾混在 evaluator | 调试职责移入 diagnostics，map memory 推理能力保留。提交：`22ee635` |
| B05 | SatNav 脚本公共 I/O | B | 四个脚本重复 read/write episode、annotation、summary、copy tree、reindex 等 helper | 若保留脚本，提取 `satnav_io.py`；若归档一次性脚本，则只为现行脚本抽取 |
| B06 | S2R 数据生成小工具重复 | B | SUES/UAVVisLoc 的参数解析完全一致；DenseUAV/GTA 的裁剪/NCC 逻辑相似 | 只做共享 image/pair utility，不删除近期 Stage-A 工作流 |
| B07 ✅ 已完成 | 单实现 Base 层 | B/C | 四组 Base 均只有一个 SwiftVLN 实现 | 训练类合并到具体实现；评测改为 inference、environment、result recorder 等组合组件。提交：`c1e61e9`、`0cd6bd1`、`cfc4a8b`、`3f5dadf` |
| B08 ✅ 已完成 | `common` 顶层大面积 lazy re-export | B/C | 顶层 facade 隐藏真实依赖 | 内部改为具体模块 import，`common.__all__` 收缩为空。提交：`f8cd0fe` |
| C01 ✅ 已完成 | 三个 2026-04-21 的一次性 SatNav split 脚本 | C | 共 2,127 行，绑定历史数据版本且无主线引用 | 按维护者确认直接删除。提交：`da2df3c` |
| C02 ✅ 已完成 | R2R/UniNaVid ViT 预计算工具 | C | Python+shell 共 455 行，无主线消费者 | 按维护者确认直接删除。提交：`da2df3c` |
| C03 ✅ 已完成 | 旧 trajectory frame 分析工具 | C | 脚本、README、生成报告合计 747 行 | 按维护者确认直接删除。提交：`da2df3c` |
| C04 ✅ 已完成 | `analyze_satnav_results.py` | C | 读取旧 JSON 数组，与当前 JSONL 不兼容 | 按维护者确认直接删除。提交：`da2df3c` |
| C05 ✅ 已完成 | Habitat 未使用 measure 与 `maps.py` | C | 当前 YAML 只配置 OracleSuccess/OracleNavigationError | 删除 `maps.py` 和其余兼容 measure；两种 backbone 的 Habitat 实流验证通过。提交：`af22dc1` |
| C06 ✅ 已完成 | 整个 deployment 子系统 | C | 子系统与当前模型路径和能力脱节 | 按维护者决定删除 deployment、CLI deploy、runner 和启动脚本，共删除 1,714 行。提交：`c1cb769` |
| C07 ✅ 已完成 | CLI registry/runners 的 train/eval/queue | C | registry 只有 `swiftvln` 一个 key，规范入口绕过 runner | 删除 registry/runners；CLI 直接调用唯一的 train/eval 实现，稳定 shell 入口保留。提交：`5a871f6` |
| C08 ✅ 已完成 | train/eval queue 的交互式向导 | C | 自动化已有文件配置/todo 协议 | 删除 wizard、shortcode 和 positional 模式；训练只接受 `TRAIN_EXPERIMENTS_FILE`，评测只接受 todo 文件。提交：`622c1bc`、`84b8e9c` |
| C09 | `s2r/__init__.py` 历史顶层导出 | C | 文件明确称为 historical top-level exports；内部代码主要直接 import 子模块 | 确认是否承诺 Python public API，再决定移除兼容层 |
| D01 | GTC/SegmentGTC/map/initial/pose/overlap/no-memory | D | 现有 SwiftVLN model zoo 直接覆盖 | 保留并纳入回归矩阵 |
| D02 | Qwen2.5/Qwen3 wrapper/config 注册 | D | HF checkpoint 配置和 Qwen3 inputs-embeds patch 依赖 | 类名、model_type、config type 必须保持兼容 |
| D03 ✅ 已精简 | 当前 SatNav 主线数据脚本 | D | 公开流程只需要检查数据并由 canonical episodes 生成 split | 保留 `inspect_data.py`、`run_all.py`、`process_episodes.py`；删除历史类型迁移和内部版本合并工具 |
| D04 | S2R Stage-A 训练、评测与数据生产 | D | 当前正式 pipeline，且近期新增 | 保留，避免按“非 VLN 主模型”误判为历史代码 |
| D05 | JSONL 评测恢复、去重、汇总逻辑 | D | 当前全部模型结果使用 `all_results.jsonl` | 保留并补结果恢复测试 |

## 5. 重点问题详解


下面意见就是我的意见。你可以按照现在的情况以及我的意见进行重构以及删除，不过每次修改一个前后都需要进行测试，保证修改不影响功能。测试之后的测试产物需要删除。每一次完成一项都在这个文档对应的条目下进行更新，同时改动创建commit

### 5.1 单模型仓库的多模型壳层

`common/registry.py` 的 `MODEL_TO_PACKAGE` 只有：

```python
{"swiftvln": "swiftvln.model"}
```

但 `cli.py` 的 train/eval/deploy 仍要求用户传入 `--model swiftvln`，随后 `runners/train.py` 和 `runners/eval.py` 动态 import 唯一模块。当前训练 shell 直接执行 trainer，评测 shell 直接执行 `swiftvln.model.eval`，没有经过这层。

同时，`common/training` 和 `common/eval` 的注释仍描述 StreamVLN、CompressVLN、MonoVLN 等共享场景，但仓库内只有一个具体子类。baseline 位于其他包，也不继承这些类。

推荐目标：

- 如果未来 6–12 个月没有第二个主线模型进入 `src/swiftvln`，删除 registry 的动态性；
- 将真正共享的报告、环境包装、视频和错误分析保留为组合组件；
- 将只有一个实现的 Base 类并回具体 SwiftVLN 类；
- 入口 shell 路径可先保留，内部实现逐步变薄，避免破坏技能和已有自动化。

意见：
本repo将会只会服务于swiftvln，也默认使用swiftvln。相关的冗余设计可以进行清理

实施更新（✅ 已完成）：

- 删除 `common/registry.py` 和 `runners/` 中 train/eval/queue/deploy 的动态分发；`swiftvln train`、`swiftvln eval` 现在直连唯一实现，不再要求 `--model swiftvln`。
- 合并 `BaseVLNTrainArguments`/`BaseVLNSft` 到具体训练实现；移除 `BaseVLNEval`/`BaseVLNEvaluator`，把仍然真正共享的能力改为组合组件。
- `common/__init__.py` 不再跨包 lazy re-export，内部 import 显式指向实际模块。
- 契约覆盖 CLI 参数转发、唯一模型入口、Base 类消失和直接 import；最终 57/57 单测通过。
- 对应提交：`5a871f6`、`c1e61e9`、`0cd6bd1`、`cfc4a8b`、`3f5dadf`、`f8cd0fe`。

### 5.2 配置与模型名不是单一事实源

当前至少有以下配置来源：

1. `model/arguments.py`：训练 dataclass；
2. `model/eval.py`：手写 argparse；
3. `model/script/train/*.sh`：默认值、合法值和模型名生成；
4. `model/script/eval/*.sh`：另一套默认值和拼接；
5. `scripts/eval/eval_by_name.sh`：用 sed/grep 解析名称；
6. `deployment/model_resolver.py`：独立正则和能力白名单；

已经观察到的漂移包括：

- 训练主线默认 SatNav，而底层评测 shell 文档/默认仍以 Habitat 为起点；
- 训练参数帮助文本没有完整反映 SegmentGTC；
- deploy parser 要求长模型名中的 `b<log_base>`、batch/lr/timestamp 形态，但 model zoo 使用短模型名；
- deploy 明确拒绝 map/GTC/SegmentGTC/initial/embedding；
- deploy 只查找 `output/swiftvln/<name>/checkpoint-*`，当前保留模型位于 `output/model_zoo/*/HF_model/<name>`；
- GTC 评测 summary 记录 output token 和 temperature，却漏掉 `gtc_num_iterations`。

建议建立 `SwiftVLNExperimentSpec`：

- Python dataclass/validated model；
- `parse_model_name()` 同时接受历史长名和当前短名；
- `to_model_name(style="short|run")`；
- 统一 map/overlap/history/embed 约束；
- 训练、评测、deploy、结果汇总共同调用；
- shell 通过一个 Python 子命令输出 `KEY=VALUE`，不再维护 sed/grep 规则。

迁移前必须把 11 个 SwiftVLN model-zoo 名称、3 个 backbone 名称及若干历史长名固化成 golden tests。

意见：这边确实非常乱，你需要进行修改修正。一些功能可以进行简化

实施更新（✅ 已完成）：

- 新增 `src/swiftvln/experiment.py`，以 `SwiftVLNExperimentSpec` 统一环境、backbone、memory/history、overlap、system prompt、embedding 和运行元数据的解析、验证与命名。
- `train_swiftvln_qwen_vl.sh` 通过 `build-name` 生成名称；`eval_by_name.sh`/`eval_lib.sh` 通过 `parse-name` 获取 shell 变量；结果汇总直接调用同一 Python parser，不再各自维护 sed/grep/正则规则。
- golden tests 覆盖现有 model-zoo 名称、历史长名、Qwen2.5/Qwen3、map/GTC/SegmentGTC、非默认 GTC 参数、overlap 与 round-trip。
- 外部 embedding 选择收口为 `none`、`pose`、`posefilm`、`uav` 四种互斥模式；pose-additive + UAV 和 pose-film + UAV 会在名称解析、模型加载和运行时配置阶段提前拒绝。
- 对应提交：`332cff6`、`bc1c3aa`。

### 5.3 评测器承担了过多职责

`model/evaluator.py` 有 1,725 行，当前同时负责：

- 视觉编码与压缩；
- per-frame/GTC/SegmentGTC/map history；
- prompt token/embed 构建；
- overlap/window 状态机；
- environment episode loop；
- landmark failure 分析和图片/JSON 报告；
- map prompt/tensor debug；
- timing/debug 输出。

建议先按职责拆分，而不是直接重写算法：

```text
SwiftVLNEvaluator
├── InferenceSession        # frame、prompt、history、window
├── EnvironmentEpisodeLoop # reset/step/stop/error
├── ResultRecorder          # JSONL、metric、resume
└── DiagnosticsObserver     # landmark/map/timing，可选
```

这样部署若保留，可直接复用 `InferenceSession`，无需复制 evaluator 私有方法；诊断功能也不会继续撑大主线类。

意见：可以进行职能拆分，不过拆分前后都需要进行测试

实施更新（✅ 已完成）：

- 原 1,720 行 `model/evaluator.py` 已降为 288 行的薄编排器；窗口推理移入 `model/inference.py`，命令入口移入 `model/eval_runner.py`，landmark/map/timing 诊断移入 `model/diagnostics.py`。
- 环境 episode 能力组合到 `common/eval/environment.py`，JSONL resume/去重/汇总写入集中到 `common/eval/results.py`。
- 保留原窗口、overlap、history processor、异常 episode 持久化、分布式 rank 完成标记和汇总 schema；不重写模型算法。
- 拆分前建立 window/JSONL 契约，拆分后通过 57 个单测及四组真实评测；两组 8-rank SatNav 结果均完整、无重复。
- 对应提交：`22ee635`、`0cd6bd1`、`cfc4a8b`、`3f5dadf`。

### 5.4 deployment 已经与当前仓库状态脱节

`DEFAULT_SWIFTVLN_DEPLOY_MODEL_NAME` 指向一个带时间戳的历史长名。resolver：

- 仅支持 baseline per-frame + pool + noembed；
- 明确拒绝 map、GTC、SegmentGTC、initial；
- 只接受本地 `checkpoint-*` 布局；
- 不支持 model zoo 的 `HF_model` 目录或显式 HF path。

另一方面，`deployment/policy.py` 与 evaluator 有至少 14 个同名方法，重复维护 frame encode、pose、history cache、prompt 和 overlap。

因此 deployment 需要一个明确产品决策：

- **保留**：先支持显式 `--model-path` 和 model-zoo/HF 路径，再抽取共享 inference session，最后扩展能力；
- **不保留**：一起移除 CLI deploy、`runners/deploy.py`、`deployment/` 和 `scripts/deploy/`，可减少约 1,633 行范围；
- 不建议维持当前“有入口、有文档，但默认无法解析现有模型”的状态。

意见：deployment部分暂时先删除，当前代码库先不需要deployment的feature

实施更新（✅ 已完成）：

- 删除 `deployment/`、`scripts/deploy/`、`runners/deploy.py`、CLI deploy 子命令和相关上下文说明，共删除 1,714 行。
- 删除前先用 CLI/import 契约固定保留入口；删除后确认 train/eval/queue/s2r-data 不受影响，仓库内无 deployment 残余引用。
- evaluator 的推理拆分只服务当前训练/评测需求，没有为已删除产品能力新增抽象。
- 对应提交：`c1cb769`。

### 5.5 Habitat 历史代码

`habitat_extensions/measures.py` 后半段保存了被整段注释的旧实现；这些注释还使 gzip/json/pickle、logger、Action、fog_of_war、地图工具等 import 看似必要。`maps.py` 的活跃函数没有被当前主线调用，现有 Habitat YAML 只配置 `OracleSuccess` 和 `OracleNavigationError`。

建议分两步：

1. 先删除明确的注释代码和随之无用的 import；
2. 在 Habitat `vln_r2r` smoke 后，确认是否只保留两个在用 measure。若外部配置仍可能引用 PathLength/OracleSPL/StepsTaken，则保留兼容类但不必保留未使用 maps。

意见：同意按照现在的进行测试，修改之后也需要跑一下habitat的运行，看看eval是否可以正常运行，可以shiyongswiftvln的模型，只是跑一下流程

实施更新（✅ 已完成）：

- 删除不在当前 YAML 中使用的 `maps.py`、PathLength/OracleSPL/PL/StepsTaken 等兼容实现；只保留 `OracleSuccess` 和 `OracleNavigationError` 及其注册。
- 新增 Habitat 扩展契约，确认导出面与 YAML 一致。
- Qwen2.5 和 Qwen3 均完成 Habitat 数据 8-GPU、1-step 训练，并各自生成完整 checkpoint；两种 backbone 又分别完成 `vln_r2r_smoke.yaml`、`val_unseen` 的真实环境评测，均为 1 episode、5 steps、0 执行异常。
- 对应提交：`af22dc1`；实机结果见 12.3–12.4。

### 5.6 历史数据与分析工具

下列工具的“无主线引用”并不等于没有研究价值，但它们不适合继续放在可安装主包里：

- `append_eval_split_and_trajectory_to_train.py`；
- `sample_eval_split_from_train.py`；
- `build_similarity_val_seen_from_train.py`；
- `vit_feat_precompute/`；
- `analyze_trajectory_frames.py` 及其 README/生成报告；
- 与当前 JSONL 不兼容的 `analyze_satnav_results.py`。

建议统一迁到 `tools/archive/<topic>/` 或 `.codex/archive/code/`，附一页 README 记录输入数据版本、依赖和最后验证日期。归档不等于立即永久删除，能先把主包边界变清楚。

这部分若全部移出 `src/swiftvln`，涉及约 3,730 行，但其中 2,127 行历史 SatNav split 工具是否需要复现必须先确认。

意见：这些都可以进行删除，目前不需要这些功能

实施更新（✅ 已完成）：

- 按确认直接删除三个历史 SatNav split 脚本、ViT 预计算工具、trajectory frame 分析脚本/README/生成报告，以及旧 JSON 结果分析器，共 9 个文件、3,730 行。
- 删除后检查当前 SatNav skill 依赖的 `inspect_data.py`、`run_all.py`、`process_episodes.py` 均仍存在；关键数据和 S2R 入口 `--help` 通过。
- 没有移动或修改正式数据集、模型库和 Stage-A 主线。
- 对应提交：`da2df3c`。

### 5.7 队列脚本过重

`train_queue.sh` 1,084 行，`eval_queue.sh` 932 行。当前自动化已经有：

- `TRAIN_EXPERIMENTS_FILE` 非交互训练；
- `AUTO_TODO` / `DYNAMIC_TODO` 评测 worker；
- 独立 enqueue 脚本。

剩余大量行用于交互式 wizard、shortcode 展开、彩色摘要和多种输入格式。若团队现在只通过计划编排或 todo 队列运行，建议保留单一非交互协议，shell 只负责环境与进程启动，解析和状态管理迁到 Python。

这里不能仅靠静态引用判断，因为交互入口可能由人直接使用。需要维护者确认最近一个月是否仍有人手工运行向导。

意见：这边整体可以进行精简，因为后续应该还是使用config的形式来进行启动，不需要再进行手工运行向导。所以这边交互的部分可以删除。同样做好本职的测试

实施更新（✅ 已完成）：

- 训练队列只接受 `TRAIN_EXPERIMENTS_FILE` 的严格五字段 pipe 配置；删除交互 wizard、shortcode 和宽松多格式解析，并新增 `--check-config` 无启动校验。
- 评测队列只消费 todo 文件；删除位置参数模型列表和交互向导，保留 `DYNAMIC_TODO`/`WAIT_FOR_NEW_TASKS` worker 行为及 `AUTO_TODO` 兼容映射，并新增 `--check-queue`。
- 新增 9 个队列协议测试，覆盖合法配置、非法名称、只读检查、动态等待约束、旧位置参数拒绝和“校验不触发真实任务”。
- 后续开源精简进一步删除了内部训练 sidecar、事件协议和额外 CSV 汇总器；队列继续以 host 状态 JSON 与每次评测的 `evaluation_summary.json` 为结果边界。
- 对应提交：`622c1bc`、`84b8e9c`。

## 6. 建议的目标结构

建议：我觉得很ok

实际落地后的核心结构如下：

```text
src/swiftvln/
├── cli.py                         # 单 SwiftVLN 公共命令，直接调用具体实现
├── experiment.py                  # ExperimentSpec + validation + name codec
├── model/
│   ├── arguments.py               # 具体训练参数
│   ├── trainer.py                 # 具体训练实现
│   ├── eval.py / eval_runner.py   # CLI 与分布式评测编排
│   ├── evaluator.py               # 薄 episode orchestration
│   ├── inference.py               # 窗口、prompt、history 推理状态
│   ├── diagnostics.py             # landmark/map/timing 可选诊断
│   ├── model.py                   # HF config/model/loader 注册
│   ├── dataset.py
│   ├── template.py
│   └── map_memory.py
├── common/
│   ├── env/                       # Habitat/SatNav 数据与动作 adapter
│   ├── eval/environment.py        # 环境能力组合
│   ├── eval/results.py            # JSONL、resume、去重、汇总
│   ├── history_processors/        # per-frame/GTC/SegmentGTC/map 支撑
│   ├── embedding_enhancement/
├── s2r/                           # 当前 Stage-A 主线
└── scripts/                       # 稳定薄入口和非交互配置/todo 队列
```

`deployment/`、`runners/`、历史分析/数据脚本已经移除。关键原则仍是按“当前是否有多个实现”决定抽象；未来增加数据或环境时扩展明确的参数、dataset/env adapter 和 environment composition 边界，而不是恢复全局 registry/Base 壳层。

## 7. 分阶段执行方案

### Phase 0：建立防护网（✅ 已完成）

在任何结构删除前先补：

1. model-name parser golden tests：11 个 SwiftVLN、3 个 backbone、代表性历史长名；
2. history sampling/compression 固定输入测试；
3. GTC 与 SegmentGTC soft-kmeans 等价测试；
4. overlap/window 状态机测试；
5. pose action reconstruction 测试；
6. JSONL resume、去重和异常 episode 测试；
7. map 配置约束测试；
8. CLI/import/help smoke。

完成结果：新增 15 个测试文件、57 个 CPU 契约测试，覆盖名称/config、history、pose、embedding、window、JSONL、CLI、队列、训练/评测结构和 Habitat 扩展；提交 `d0b071c` 及各阶段随附测试。

### Phase 1：无行为精简

**状态：✅ A01–A07 已完成。** 本阶段只处理候选总表中的 A01–A07；下列更大范围的历史工具、deployment、配置和队列精简尚未借此提交提前实施。

建议一个或数个小提交完成：

- 删除 Habitat 注释块和确定无用 import；
- 修复 Ruff 的安全项，跳过有意的 EGL E402；
- 删除未使用变量、`episode_exception`；
- 删除 deprecated `get_compressed_length`；
- 删除 pose reconstruction 的 no-op `norm_scale`；
- 删除 `run_all --episodes-only`；
- 合并 GTC/SegmentGTC 重复 soft-kmeans；
- 合并 Qwen loader 初始化 helper；
- 收窄异常捕获。

验证：compileall、Ruff、全部新增单测、主入口 `--help`、模型 config/load smoke。

### Phase 2：历史工具清理（✅ 已完成）

- 维护者明确这些历史能力不再需要，因此没有保留虚假归档入口，直接删除 9 个文件、3,730 行；
- 当前 SatNav skill 使用的脚本、S2R Stage-A 入口和配置全部保留；
- 数据与 S2R 入口 smoke 通过。提交：`da2df3c`。

### Phase 3：统一配置和名称解析（✅ 已完成）

- 引入 `SwiftVLNExperimentSpec`，统一 train/eval/map/history/embed/overlap/backbone 校验；
- 训练名称生成、`eval_by_name.sh`、eval helper 和结果汇总都迁移到统一 codec；
- 所有 model-zoo/历史 golden names 保持含义，当前训练命名保持稳定；
- deployment resolver 随整个 deployment 删除。提交：`332cff6`、`bc1c3aa`。

### Phase 4：拆分 evaluator，处理 deployment（✅ 已完成）

- 提取 inference session、diagnostics、environment composition 和 result recorder；
- 删除 deployment；删除单实现 `BaseVLNEvaluator`/`BaseVLNEval`；
- 契约覆盖 per-frame、GTC、SegmentGTC、map、initial、pose、overlap 和 JSONL，实机覆盖 Qwen2.5/Qwen3、Habitat/SatNav。提交：`c1cb769`、`22ee635`、`0cd6bd1`、`cfc4a8b`、`3f5dadf`。

### Phase 5：入口与队列收口（✅ 已完成）

- 仓库被确认只服务 SwiftVLN，CLI 删除唯一模型 registry/runners；
- 交互式 train/eval wizard 被确认不再需要；
- 稳定 shell/队列路径保留，内部只保留训练配置文件与评测 todo 文件两种非交互协议；
- `B05`、`B06` 的小工具去重和 `C09` 的 S2R 顶层兼容层没有足够收益，本轮保持不变。提交：`5a871f6`、`622c1bc`、`84b8e9c`、`f8cd0fe`。

## 8. 每阶段验证门禁

| 层级 | 门禁 |
|---|---|
| 静态 | `ruff` 无新增问题；`compileall`；全部 shell `bash -n` |
| 单元 | config/name、history processor、pose、window、JSONL tests |
| 加载 | 所有保留 HF config/model type 可注册；Qwen2.5/Qwen3 loader smoke |
| CPU 轻量 | CLI help、数据脚本 dry-run/fixture、结果汇总 fixture |
| GPU smoke | SwiftVLN train → checkpoint → eval；至少 baseline 后再覆盖高风险变体 |
| 环境 smoke | SatNav 与 Habitat 各一条最小 episode |
| 回归 | 模型名、输出目录、`all_results.jsonl` schema、队列文件路径保持兼容 |

## 9. 六个维护者决策的落实状态

1. **deployment：已决定当前不需要。** 已整体删除，不保留半可用入口。
2. **Habitat：持续支持。** 只删除未使用扩展，训练和评测主能力保留并完成双 backbone 实机验证。
3. **train/eval queue 交互向导：不再使用。** 已收口为配置文件/todo 文件协议。
4. **`ver_260418/val_seen_update` 历史重建：不再需要。** 三个一次性 split 工具已删除。
5. **Python public API：`common` 不承诺聚合 facade。** 内部已直接 import；`s2r/__init__.py` 的历史导出暂时保留，避免无收益兼容破坏。
6. **第二个主线模型：不计划加入。** registry、runners 和单实现 Base 类已精简；Qwen2.5/Qwen3 被视为同一 SwiftVLN 的 backbone 选择。

最终边界是：保留已发布模型和当前训练/评测所需能力，删除维护者明确放弃的产品/历史功能；每个决策独立提交并配套契约或实机验证。

## 10. Phase 1 实施与验证记录

初始审计阶段没有改动源码；在维护者确认按报告继续后，本轮只实施 A01–A07。结论仍遵守“无仓库内引用不等于无仓库外用户”的边界，因此没有顺带处理 B01–D05。

### 10.1 变更范围

- 修改 12 个源码文件，90 行新增、670 行删除，净减少 580 行；没有新增、删除或移动文件。
- `habitat_extensions/measures.py` 从 630 行降至 194 行；保留 `PathLength`、`OracleNavigationError`、`OracleSuccess`、`OracleSPL`、`PL`、`StepsTaken`。
- 保留 SwiftVLN Qwen2.5/Qwen3 的 config、wrapper、loader 类名与 `model_type`，以及 Qwen3 inputs-embeds patch。
- 保留 SatNav、Habitat、per-frame、GTC、SegmentGTC 的现有入口和能力。

### 10.2 验证结果

| 验证 | 结果 |
|---|---|
| Python 编译 | `python3 -m compileall -q src/swiftvln` 通过 |
| shell 语法 | `src/swiftvln` 下全部 shell `bash -n` 通过 |
| 静态检查 | Ruff 总问题 87 → 70；F841 5 → 0；Habitat measures 的 F401/F841 为 0 |
| 既有测试 | `test_uav_adapter_strategy.py` 通过 |
| 定向契约 | GTC 数值等价、GTC/SegmentGTC shape、compressor/pose/loader/wrapper/reload 契约全部通过 |
| 数据脚本 | `run_all --help` 通过，已不再暴露无效 `--episodes-only` |
| 模型加载 | Qwen3 2B 实际 loader/model smoke 通过，wrapper 与 patch 均正确挂载 |
| SatNav GPU smoke | 2 GPU、1 step 训练成功并生成完整 checkpoint；随后 `val_seen`、`val_unseen` 各完成 10 episodes |
| Habitat GPU smoke | 2 GPU、`vln_r2r_smoke.yaml`、`val_unseen` 1 episode 完整通过；NE 7.9608，平均 5 steps |

SatNav smoke 的训练 loss 为 1.29994798；`val_seen` 完成 10 episodes（NE 209.6784，平均 170 steps），`val_unseen` 完成 10 episodes（NE 19.6654，平均 16.5 steps）。这是一轮代码路径 smoke，不用于判断模型质量。

### 10.3 测试产物清理

已删除本轮专用实验名下约 57 GB 的临时 checkpoint、全部 SatNav/Habitat 评测结果、4 份 `/tmp` 日志，以及测试生成的 `__pycache__`/`.pyc`。清理后对应路径均不存在、源码树中的 `.pyc`/`__pycache__` 计数均为 0，GPU 0/1 显存已释放。正式模型库和数据集没有改动；这些临时产物未保留备份，但可通过同一 smoke 流程重建。

### 10.4 原子提交

| 条目 | 提交 | 内容 |
|---|---|---|
| A01 | `d352994` | 清理 Habitat 历史注释实现 |
| A02 | `b4619f4` | 移除无效模型缓存状态 |
| A03 | `3e8b2a7` | 删除废弃与无效接口 |
| A04 | `3d5079d` | 清理未使用局部状态 |
| A05 | `f808ca5` | 复用 GTC 聚类步骤 |
| A06 | `5bd5707` | 复用 Qwen 模型加载参数 |
| A07 | `7d3ed13` | 显式处理模型注册错误 |

## 11. 关键证据索引

以下行号记录的是初始审计证据；A01–A07 完成后部分行号或符号已不再存在，后续应以提交和符号搜索为准。

| 结论 | 位置 |
|---|---|
| CLI 对唯一模型仍要求 `--model` | `src/swiftvln/cli.py:17-24` |
| registry 只有一个 package | `src/swiftvln/common/registry.py:10-24` |
| 规范训练 shell 绕过 runner | `src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh:642` |
| 规范评测 shell 绕过 runner | `src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh:414` |
| 四组 Base 类及唯一子类 | `src/swiftvln/common/training/arguments.py:15`、`base_sft.py:23`、`common/eval/evaluator.py:51`、`common/eval/runner.py:72`；对应子类见 `model/arguments.py:14`、`trainer.py:20`、`eval.py:38`、`evaluator.py:131` |
| KV-cache mixin | `src/swiftvln/model/model.py:27-112`；仓库内只发现 reset 调用在 `common/eval/runner.py:333` 和 `deployment/policy.py:99-101` |
| deprecated compressor API | `src/swiftvln/common/history_processors/compressor.py:358-376` |
| pose no-op 参数 | `src/swiftvln/common/embedding_enhancement/pose_utils.py:16-33` |
| `episodes_only` 不参与分支 | `src/swiftvln/scripts/data_process/run_all.py:24-27,97-116` |
| GTC/SegmentGTC 重复方法 | `src/swiftvln/common/history_processors/gtc.py:127-143`、`segment_gtc.py:202-218` |
| Habitat 大段注释代码 | `src/swiftvln/habitat_extensions/measures.py:221-630` |
| 当前 Habitat YAML 只用两个扩展 measure | `src/swiftvln/configs/vln_r2r.yaml:43-45`、`vln_r2r_smoke.yaml:43-45` |
| landmark debug 主体 | `src/swiftvln/model/evaluator.py:1197-1400` |
| map debug 主体 | `src/swiftvln/model/evaluator.py:532-679`、`src/swiftvln/model/map_memory.py:61-81,306-333` |
| evaluator/deploy 重复窗口推理 | `src/swiftvln/model/evaluator.py:115-129,338-1186`、`src/swiftvln/deployment/policy.py:30-41,222-501` |
| deploy 只支持部分名字/能力 | `src/swiftvln/deployment/model_resolver.py:13-40,131-187` |
| 当前结果写入 JSONL | `src/swiftvln/common/eval/runner.py:589` |
| 旧分析器读取 JSON 数组 | `src/swiftvln/scripts/analysis/analyze_satnav_results.py:342,368` |
| 训练队列非交互入口 | `src/swiftvln/scripts/train/train_queue.sh:411-434` |
| 评测队列非交互入口 | `src/swiftvln/scripts/eval/eval_queue.sh:762-840` |

当前实现不再使用上述已删除路径和旧行号。重构后的关键证据入口为：

| 结论 | 当前位置 |
|---|---|
| 单模型 CLI 直连具体实现 | `src/swiftvln/cli.py` |
| 配置、校验和模型名唯一事实源 | `src/swiftvln/experiment.py` |
| 训练具体参数与实现 | `src/swiftvln/model/arguments.py`、`src/swiftvln/model/trainer.py` |
| 推理 session / 薄 evaluator / CLI runner | `src/swiftvln/model/inference.py`、`evaluator.py`、`eval_runner.py` |
| 环境组合与结果记录 | `src/swiftvln/common/eval/environment.py`、`results.py` |
| 独立诊断 | `src/swiftvln/model/diagnostics.py` |
| 互斥 embedding 运行时边界 | `src/swiftvln/common/embedding_enhancement/runtime.py`、`src/swiftvln/model/model.py` |
| 非交互队列协议 | `src/swiftvln/scripts/train/train_queue.sh`、`src/swiftvln/scripts/eval/eval_queue.sh` |
| 重构契约测试 | `tests/test_*contract*.py`、`tests/test_*protocol.py` |

## 12. 全量重构实施与验收记录

### 12.1 最终变更范围

- 从重构契约基线到最终代码，共 92 个代码/测试/上下文文件发生变化，新增 5,867 行、删除 10,981 行，净减少 5,114 行；其中 `src/swiftvln` 相对该基线净减少约 6,424 行。
- 相对最初审计口径，`src/swiftvln` 从 141 个文件、33,272 行降至 114 个文件、26,269 行，净减少 27 个文件、7,003 行。
- 删除 33 个源码文件，新增 6 个职责明确的源码文件；新增内容主要是统一 schema、推理 session、diagnostics、环境组合和结果记录，而不是重复功能分支。
- `.codex/CODEX_CONTEXT.md` 已随重大结构变化同步更新；训练/评测稳定脚本路径、数据默认约定和结果 JSONL 边界保持有效。

### 12.2 静态与契约门禁

| 门禁 | 最终结果 |
|---|---|
| 编译 | `python -m compileall -q src/swiftvln tests` 通过 |
| shell | `src/swiftvln` 下全部 `.sh` 执行 `bash -n` 通过 |
| Ruff | 初始 87 个问题 → **0**，`src/swiftvln tests` 全部通过 |
| 单元/契约 | **57/57** 通过，运行时间 2.666 秒 |
| UAV adapter | 在主线训练环境完成外部 checkpoint 加载、挂载和 forward shape smoke |
| 入口 | CLI、SatNav 数据脚本和 S2R manifest 的 `--help` smoke 通过 |
| JSON 结果 | 四组结果的 count、唯一 episode key、model type、world size、rank 集合和异常字段自动断言通过 |

### 12.3 8-GPU 训练矩阵

四组训练都在本机服务器 98 的 8 张 H100 上执行，限制为 16 个样本、`max_steps=1`，并验证最终 checkpoint 的 config 和权重完整性。

| Backbone | 训练环境/数据 | 结果 | loss | checkpoint |
|---|---|---|---:|---|
| Qwen2.5-VL 3B | SatNav | 通过 | 1.27842259 | 完整，两份 safetensors shard |
| Qwen2.5-VL 3B | Habitat（R2R/RxR） | 通过 | 1.35145164 | 完整，两份 safetensors shard |
| Qwen3-VL 2B | SatNav | 通过 | 1.63227499 | 完整，单份 safetensors；token acc 0.68604651 |
| Qwen3-VL 2B | Habitat（R2R/RxR） | 通过 | 1.69761992 | 完整；token acc 0.63141994 |

四组任务都正常退出并释放 8 张 GPU。日志中的 NCCL “process group 未显式 destroy”是成功退出后的既有 warning，没有造成残留进程或 checkpoint 不完整。

### 12.4 SatNav/Habitat 评测矩阵

SatNav 使用 8 GPU、每卡 1 episode 验证分布式收集；Habitat 使用单模拟器、1 episode 验证稳定实流。每个 episode 最多 5 steps，关闭视频和自动恢复。

| Backbone | 评测环境 | world size / episodes | SR / SPL / OS | NE | 平均 steps | 异常 |
|---|---|---:|---|---:|---:|---:|
| Qwen2.5-VL 3B | SatNav `val_seen` | 8 / 8 | 0 / 0 / 0 | 151.131708 | 4.75 | 0 |
| Qwen2.5-VL 3B | Habitat `val_unseen` | 1 / 1 | 0 / 0 / 0 | 7.960824 | 5.00 | 0 |
| Qwen3-VL 2B | SatNav `val_seen` | 8 / 8 | 0 / 0 / 0 | 143.782667 | 5.00 | 0 |
| Qwen3-VL 2B | Habitat `val_unseen` | 1 / 1 | 0 / 0 / 0 | 7.226479 | 5.00 | 0 |

两组 SatNav 的 `result.jsonl` 均恰好包含 rank 0–7，`scene_id::episode_id` 无重复；两组 Habitat 的 dataset、Habitat-Sim、task、模型模板、推理 session、environment step、JSONL 和 summary 全链路完成。这些是 1-step checkpoint 的代码路径 smoke，指标不代表模型质量。

### 12.5 Habitat 并发说明

第一次尝试用 8 个 Habitat-Sim 进程并发评测 Qwen2.5 时，3 个 rank 正常完成，另外 5 个 rank 停在 NVIDIA/Habitat-Sim 驱动读写锁，同时 `nvidia-smi` 短暂阻塞；日志中没有 Python/模型异常。该专用 tmux 和子进程被精确终止后驱动恢复，8 张卡显存全部释放。随后使用单 Habitat 模拟器稳定完成 Qwen2.5 和 Qwen3 两组实流。

因此本轮结论是“SwiftVLN 的 Habitat 功能路径通过”，而不是“当前机器上的 8-way Habitat-Sim 并发已得到保证”。若未来把多模拟器并发作为生产要求，需要单独做驱动/Habitat-Sim 稳定性验证。

### 12.6 测试产物清理

- 删除专用 `runtime/smoke/refactor_final_matrix_20260805`：约 **178 GB、203 个文件**，包含四组临时 checkpoint 和全部日志。
- 删除五个带本轮唯一时间戳的评测模型目录：四个成功结果和一个并发中断的 partial result。
- 删除 `src/`、`tests/` 下测试生成的全部 `__pycache__`/`.pyc`；最终计数为 0。
- 清理后服务器 98 的 GPU compute process 计数为 0；所有本轮专用 tmux 均不存在，用户原有 `rebuttal` session 未触碰。
- 正式模型库、数据集、正式评测结果和用户已有运行任务均未删除。上述临时产物按维护者要求不保留备份，可由同一 smoke 配置重建。

### 12.7 本轮原子提交

| 提交 | 内容 | 主要门禁 |
|---|---|---|
| `d0b071c` | 建立 SwiftVLN 重构契约基线 | config/name/history/window/JSONL/CLI 契约 |
| `c1cb769` | 移除本地 deployment 功能 | CLI/import 契约 |
| `da2df3c` | 删除历史数据与分析工具 | 当前数据/S2R 入口 smoke |
| `af22dc1` | 收缩 Habitat 扩展兼容面 | Habitat 扩展契约 + 实流 |
| `5a871f6` | 收口单模型 CLI 入口 | CLI 转发/help 契约 |
| `332cff6` | 统一实验配置与模型命名 | golden names + shell/collector 契约 |
| `22ee635` | 拆分评测推理与环境循环 | window/JSONL 契约 |
| `c1e61e9` | 合并单实现训练 Base 层 | 训练结构契约 + GPU train |
| `0cd6bd1` | 抽离评测结果记录组件 | resume/去重/summary 契约 |
| `cfc4a8b` | 组合评测环境能力 | 环境分发/episode 分布契约 |
| `3f5dadf` | 移除单实现评测 Base 层 | 评测结构契约 + GPU eval |
| `622c1bc` | 收口训练队列配置协议 | 3 个队列协议测试 |
| `84b8e9c` | 统一评测文件队列协议 | 6 个队列协议测试 |
| `f8cd0fe` | 收缩 `common` 顶层导出 | import surface 契约 |
| `bc1c3aa` | 禁止组合 embedding 增强 | 4 个边界层契约 |
| `7eb9cda` | 清理剩余静态检查问题 | Ruff 0 + 全量回归 |

# 我的需求
0. 整体的代码都可以比较结构化，方便后续添加新的数据进行训练或者新的环境进行评测

1. 模型可以接受qwen2.5以及qwen3两种模型作为基础模型进行训练，同时代码结构上并不会非常丑陋

2. 模型训练的时候可以使用satnav或者habitat的数据进行训练，同时eval的时候也可以使用habitat或者satnav进行eval。训练以及eval的代码我希望可以结构化一些，后续有可能需要添加新的环境进行训练

3. 模型训练以及eval整体的代码结构都需要简单容易阅读容易评审

4. 我希望最后模型memory中的选择在外部看比较简洁， embedding层面这两个可以不要保存 (- pose-additive + UAV) - (pose-film + UAV)

5. 代码整洁，可读性非常高

## 验收映射（2026-08-05）

| 需求 | 状态 | 落地证据 |
|---:|---|---|
| 0 | ✅ | 训练参数、实验 schema、dataset/env adapter、environment composition 职责分离；新增数据或环境不需要修改全局多模型 registry |
| 1 | ✅ | `SwiftVLNExperimentSpec` 显式支持 Qwen2.5/Qwen3；两种 backbone 均完成 8-GPU 训练和 SatNav/Habitat 评测 |
| 2 | ✅ | SatNav/Habitat 均可作为训练数据和评测环境；四组训练 + 四组评测矩阵通过 |
| 3 | ✅ | 训练 Base 合并，评测拆为 inference/evaluator/environment/results/diagnostics；稳定 shell 入口保持不变 |
| 4 | ✅ | memory/history 命名与校验集中；pose-additive + UAV、pose-film + UAV 在名称、加载和运行时三层拒绝 |
| 5 | ✅ | 初始 33,272 行降至 26,269 行，Ruff 87 → 0，57 个契约测试和职责边界共同保证可读性与可评审性 |
