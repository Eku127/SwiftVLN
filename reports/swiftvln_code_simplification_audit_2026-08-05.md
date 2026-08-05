# SwiftVLN 代码精简审计报告

> 初始审计日期：2026-08-05<br>
> 审计范围：`src/swiftvln/`<br>
> 初始审计约束：只审计、分类和规划，不删除、不重构任何源码<br>
> 实施更新：2026-08-05 已完成 Phase 1 的 A01–A07 源码精简，本文档同步记录实现、验证与提交<br>
> 结论性质：以静态引用、当前入口、仓库模型库和默认事实文档为依据；涉及外部调用方的项目均列为“需确认”，不直接判死

## 1. 结论摘要

当前 `src/swiftvln` 的主要问题并不是某一个大文件，而是四类历史设计叠加：

1. **单模型仓库仍保留多模型框架抽象**：CLI、registry、runner、训练/评测 Base 类都在为不存在的第二个主线模型提供间接层。
2. **同一份实验配置存在多套事实源**：训练参数、评测参数、训练 shell、评测 shell、模型名解析、部署解析和结果汇总各自维护规则，已经发生漂移。
3. **一次性工具和调试功能长期留在主包**：历史 SatNav split 脚本、R2R/UniNaVid 特征预计算、旧分析工具、landmark/map debug 代码占用了较多主流程体积。
4. **推理、数据处理和小型算法存在重复实现**：评测与部署维护两套窗口推理，多个 SatNav 脚本重复 I/O，GTC/SegmentGTC 重复 soft k-means 步骤。

建议的精简顺序是：

- 先补契约测试和模型名 golden cases；
- 再清理确定无行为影响的注释、无用变量、废弃 API 和 no-op 参数；
- 然后归档一次性工具；
- 再统一配置与模型名解析；
- 最后处理评测/部署推理合并，以及 Base 类、CLI、队列交互层等结构性问题。

不建议为了行数直接删除以下能力：GTC、SegmentGTC、map memory、initial prompt、pose、overlap/no-memory、Qwen2.5-VL、Qwen3-VL、SatNav/Habitat。当前模型库仍保存这些变体，贸然删除会破坏已有模型复现。

实施进度：本轮已完成 A01–A07，共修改 12 个源码文件，净减少 580 行；没有改变 HF wrapper/config 类名、模型类型、主训练/评测入口或 SatNav/Habitat 能力边界。B01 及后续结构性工作仍保持未实施状态。

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

Phase 1 A01–A07 完成后，文件数不变，源码总行数为 32,692；七个源码提交合计 90 行新增、670 行删除，净减少 580 行。新增内容主要是共享 helper 和显式注册逻辑，不是新增功能分支。

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

Phase 1 实施后，Ruff 问题由 87 个降至 70 个，5 个未使用局部变量已全部清零；`compileall`、全部 shell `bash -n`、现有 UAV adapter 测试、针对本轮改动的契约测试及 Qwen3 实际模型加载均通过。完整集成验证见第 10 节。

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
| B01 | 统一实验配置和模型名 schema | B | 参数/默认值分散在 arguments、eval argparse、训练/评测 shell、`eval_by_name.sh`、deploy resolver、results collector | 建立一个 Python schema/parser/serializer；shell 只作薄包装 |
| B02 | 评测与部署的推理核心 | B/C | 两边重复 `OverlapContext`、`TurnContext`、frame encode、history cache、prompt 构造和滑窗逻辑 | 若保留部署，先提取共享 inference session；若不保留部署则不做过度抽象 |
| B03 | evaluator 内 landmark 调试分析 | B | `evaluator.py` 1197–1400 约 204 行专用于一次 landmark failure 分析，另有主流程 hook | 移到 `tools/debug` 或独立 observer，不让核心 evaluator 承担报告生成 |
| B04 | evaluator/map_memory 的 map debug | B | evaluator 约 144 行 map prompt/tensor dump；map_memory 另有 debug render 和环境变量逻辑 | 提取可选 diagnostics 模块；保留必要的 map memory 能力 |
| B05 | SatNav 脚本公共 I/O | B | 四个脚本重复 read/write episode、annotation、summary、copy tree、reindex 等 helper | 若保留脚本，提取 `satnav_io.py`；若归档一次性脚本，则只为现行脚本抽取 |
| B06 | S2R 数据生成小工具重复 | B | SUES/UAVVisLoc 的参数解析完全一致；DenseUAV/GTA 的裁剪/NCC 逻辑相似 | 只做共享 image/pair utility，不删除近期 Stage-A 工作流 |
| B07 | 单实现 Base 层 | B/C | `BaseVLNTrainArguments`、`BaseVLNSft`、`BaseVLNEval`、`BaseVLNEvaluator` 均只有一个 SwiftVLN 子类 | 若确认仓库长期单主模型，合并到具体实现或改为小型组合组件 |
| B08 | `common` 顶层大面积 lazy re-export | B/C | `common/__init__.py` 暴露训练、评测、环境、历史处理器和工具，隐藏真实依赖 | 内部改为直接模块 import，再缩小 public surface |
| C01 | 三个 2026-04-21 的一次性 SatNav split 脚本 | C | 共 2,127 行；绑定 `ver_260418/val_seen_update` 历史流程；当前 SatNav skill 和主线无引用 | 默认建议归档，确认不需要历史数据重建后再移出主包或删除 |
| C02 | R2R/UniNaVid ViT 预计算工具 | C | Python+shell 共 455 行；硬编码 R2R/RxR 路径，生成物无主线消费者 | 若仍服务 UniNaVid，迁到 baseline 工具目录；否则归档 |
| C03 | 旧 trajectory frame 分析工具 | C | 脚本、README、生成报告合计 747 行；默认 R2R/RxR 路径，无当前主线引用 | 归档到历史分析目录，生成报告不应放在 Python 主包内 |
| C04 | `analyze_satnav_results.py` | C | 401 行；读取 `all_results.json` 数组，但当前 44 份模型结果均为 `all_results.jsonl` | 若仍需要则改造成 JSONL 工具并加测试，否则归档 |
| C05 | Habitat 未使用 measure 与 `maps.py` | C | 当前两个 Habitat YAML 只配置 OracleSuccess/OracleNavigationError；`maps.py` 的活跃调用仅来自注释掉代码 | 先做 Habitat smoke，再决定只保留两个实际 measure，或保留兼容集合 |
| C06 | 整个 deployment 子系统 | C | 1,518 行，另有 115 行启动脚本；当前 resolver 只找 `output/swiftvln/.../checkpoint-*`，不能使用现有 model zoo HF 目录 | 维护者选择“修复并共用推理核心”或“整体归档”，不要继续维护半可用状态 |
| C07 | CLI registry/runners 的 train/eval/queue | C | registry 只有 `swiftvln` 一个 key；CLI 却强制 `--model swiftvln`；规范入口绕过这些 runner | 若无外部用户，删除动态 registry/runner，CLI 只保留实际公共命令 |
| C08 | train/eval queue 的交互式向导 | C | 两个队列脚本合计 2,016 行；自动编排使用 `TRAIN_EXPERIMENTS_FILE`、`AUTO_TODO/DYNAMIC_TODO` | 若人工不再交互启动，保留非交互 worker，去掉 wizard/shortcode/UI 分支 |
| C09 | `s2r/__init__.py` 历史顶层导出 | C | 文件明确称为 historical top-level exports；内部代码主要直接 import 子模块 | 确认是否承诺 Python public API，再决定移除兼容层 |
| D01 | GTC/SegmentGTC/map/initial/pose/overlap/no-memory | D | 现有 SwiftVLN model zoo 直接覆盖 | 保留并纳入回归矩阵 |
| D02 | Qwen2.5/Qwen3 wrapper/config 注册 | D | HF checkpoint 配置和 Qwen3 inputs-embeds patch 依赖 | 类名、model_type、config type 必须保持兼容 |
| D03 | 当前 SatNav 主线数据脚本 | D | `inspect_data.py`、`run_all.py`、`process_episodes.py`、`normalize_trajectory_types.py`、`merge_satnav_data.py` 被当前 skill/上下文采用 | 保留；只去重 helper 和 no-op 参数 |
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

### 5.2 配置与模型名不是单一事实源

当前至少有以下配置来源：

1. `model/arguments.py`：训练 dataclass；
2. `model/eval.py`：手写 argparse；
3. `model/script/train/*.sh`：默认值、合法值和模型名生成；
4. `model/script/eval/*.sh`：另一套默认值和拼接；
5. `scripts/eval/eval_by_name.sh`：用 sed/grep 解析名称；
6. `deployment/model_resolver.py`：独立正则和能力白名单；
7. `scripts/eval/collect_eval_results.py`：独立识别实验变体和排序。

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

### 5.5 Habitat 历史代码

`habitat_extensions/measures.py` 后半段保存了被整段注释的旧实现；这些注释还使 gzip/json/pickle、logger、Action、fog_of_war、地图工具等 import 看似必要。`maps.py` 的活跃函数没有被当前主线调用，现有 Habitat YAML 只配置 `OracleSuccess` 和 `OracleNavigationError`。

建议分两步：

1. 先删除明确的注释代码和随之无用的 import；
2. 在 Habitat `vln_r2r` smoke 后，确认是否只保留两个在用 measure。若外部配置仍可能引用 PathLength/OracleSPL/StepsTaken，则保留兼容类但不必保留未使用 maps。

意见：同意按照现在的进行测试，修改之后也需要跑一下habitat的运行，看看eval是否可以正常运行，可以shiyongswiftvln的模型，只是跑一下流程

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

### 5.7 队列脚本过重

`train_queue.sh` 1,084 行，`eval_queue.sh` 932 行。当前自动化已经有：

- `TRAIN_EXPERIMENTS_FILE` 非交互训练；
- `AUTO_TODO` / `DYNAMIC_TODO` 评测 worker；
- 独立 watchdog 和 enqueue 脚本。

剩余大量行用于交互式 wizard、shortcode 展开、彩色摘要和多种输入格式。若团队现在只通过计划编排或 todo 队列运行，建议保留单一非交互协议，shell 只负责环境与进程启动，解析和状态管理迁到 Python。

这里不能仅靠静态引用判断，因为交互入口可能由人直接使用。需要维护者确认最近一个月是否仍有人手工运行向导。

意见：这边整体可以进行精简，因为后续应该还是使用config的形式来进行启动，不需要再进行手工运行向导。所以这边交互的部分可以删除。同样做好本职的测试

## 6. 建议的目标结构

建议：我觉得很ok

这是方向示意，不要求一次性重构：

```text
src/swiftvln/
├── cli.py                         # 只保留真实公共命令
├── model/
│   ├── config.py                  # ExperimentSpec + validation + name codec
│   ├── model.py                   # HF config/model/loader 注册
│   ├── training.py                # 具体训练实现
│   ├── evaluation.py              # episode orchestration
│   ├── inference.py               # train/eval/deploy 共享的窗口推理原语
│   ├── dataset.py
│   ├── template.py
│   └── map_memory.py
├── common/
│   ├── env/                       # Habitat/SatNav adapter
│   ├── history_processors/        # per-frame/GTC/SegmentGTC
│   ├── embedding_enhancement/
│   └── reporting/                 # JSONL、video、error/timing
├── s2r/                           # 当前 Stage-A 主线
└── scripts/                       # 稳定薄入口；一次性脚本移到仓库 tools/archive
```

关键原则：按“当前是否有多个实现”决定抽象，而不是按“未来也许会有”提前构建 registry/Base 层。

## 7. 分阶段执行方案

### Phase 0：建立防护网

在任何结构删除前先补：

1. model-name parser golden tests：11 个 SwiftVLN、3 个 backbone、代表性历史长名；
2. history sampling/compression 固定输入测试；
3. GTC 与 SegmentGTC soft-kmeans 等价测试；
4. overlap/window 状态机测试；
5. pose action reconstruction 测试；
6. JSONL resume、去重和异常 episode 测试；
7. map 配置约束测试；
8. CLI/import/help smoke。

完成标准：测试可在无 GPU 环境执行，GPU smoke 只作为后续集成门禁。

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

### Phase 2：历史工具归档

- 先归档三份 SatNav split 工具；
- 归档 R2R/RxR trajectory/VIT 工具；
- 对 `analyze_satnav_results.py` 做“升级 JSONL 或归档”的二选一；
- 把生成报告移出 Python 包；
- 记录历史输入版本和复现说明。

验证：当前 SatNav skill 引用的脚本全部存在；README、skill 和 shell 中无断链。

### Phase 3：统一配置和名称解析

- 引入 `SwiftVLNExperimentSpec`；
- 统一 train/eval/map/history/embed/overlap 校验；
- 迁移 `eval_by_name.sh`；
- 迁移 model-name 生成和结果汇总；
- 最后处理 deploy resolver。

验证：所有 model-zoo 名称 round-trip；历史长名解析结果与现有 shell 一致；训练命名不变。

### Phase 4：拆分 evaluator，处理 deployment

- 提取 inference session；
- 提取 diagnostics observer；
- 让 evaluator 和部署共享 session；
- 或在决定不保留部署后整体归档 deployment；
- 随后再评估 `BaseVLNEvaluator`/`BaseVLNEval` 是否还有价值。

验证至少覆盖：per-frame baseline、GTC、SegmentGTC、map、initial、pose、overlap、Qwen3、Habitat 和 SatNav。

### Phase 5：入口与队列收口

- 确认 CLI train/eval/queue 是否有外部用户；
- 确认交互式 train/eval queue 是否仍使用；
- 保留稳定 shell 路径作为兼容 wrapper；
- 内部只保留一种非交互配置协议。

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

## 9. 需要维护者确认的六个决策

1. **deployment 是否仍是当前产品能力？** 若否，建议整体归档；若是，优先修复模型路径并共用 inference session。
2. **Habitat 是持续支持，还是只保留历史兼容？** 当前上下文仍称支持，因此本报告默认不删除 Habitat 主能力。
3. **train/eval queue 的交互向导是否仍有人使用？** 若只剩自动化，应收口为非交互协议。
4. **`ver_260418/val_seen_update` 是否还要求精确重建？** 若否，归档三个一次性 split 工具。
5. **是否承诺 Python public API？** 特别是 `swiftvln.common.*`、`swiftvln.s2r.*` 顶层导出和 CLI train/eval/queue。
6. **是否计划在 `src/swiftvln` 内加入第二个主线模型？** 若没有，registry 和单实现 Base 类应精简。

本报告的默认建议是：保留所有已发布模型所需能力；优先归档历史一次性工具；把 deployment、交互队列和 public compatibility 层作为三个独立决策，不与低风险清理混在同一提交中。

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

# 我的需求
0. 整体的代码都可以比较结构化，方便后续添加新的数据进行训练或者新的环境进行评测

1. 模型可以接受qwen2.5以及qwen3两种模型作为基础模型进行训练，同时代码结构上并不会非常丑陋

2. 模型训练的时候可以使用satnav或者habitat的数据进行训练，同时eval的时候也可以使用habitat或者satnav进行eval。训练以及eval的代码我希望可以结构化一些，后续有可能需要添加新的环境进行训练

3. 模型训练以及eval整体的代码结构都需要简单容易阅读容易评审
