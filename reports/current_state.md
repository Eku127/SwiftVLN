# SwiftVLN 当前仓库状态

最后核验：2026-08-02

## 当前结论

- 唯一 active 仓库根目录是 /mnt/data1/home/jiangjiajun/workspace/SwiftVLN。
- 主线包是 src/swiftvln，主线模型标识是 swiftvln。
- Qwen2.5-VL 与 Qwen3-VL 共用一套训练/评测入口，通过 MODEL_FAMILY 选择。
- 历史训练目录、旧评测目录、普通日志和 SwanLab 本地缓存已清理；需要长期保留的实验资产只以 output/model_zoo 为准。
- 仓库当前保留的是推理/评测模型，不保证 DeepSpeed optimizer 级断点续训。

## 代码与入口

| 用途 | 当前入口 |
| --- | --- |
| Python 包 | src/swiftvln/ |
| CLI | swiftvln（src/swiftvln/cli.py） |
| SwiftVLN 单次训练 | src/swiftvln/model/script/train/train_swiftvln_qwen_vl.sh |
| SwiftVLN 训练队列 | src/swiftvln/scripts/train/train_queue.sh |
| SwiftVLN 分布式评测 | src/swiftvln/model/script/eval/eval_swiftvln_qwen_vl_distributed.sh |
| SwiftVLN 按名称评测 | src/swiftvln/scripts/eval/eval_by_name.sh |
| SatNav 环境配置 | src/swiftvln/configs/satnav_task.yaml |
| S2R 数据生产 | swiftvln s2r-data；实现位于 src/swiftvln/s2r/data_generation/ |
| S2R Stage-A | src/swiftvln/s2r/trainer.py 与 src/swiftvln/s2r/eval.py |
| 安装说明 | docs/installation.md |

项目元数据当前为 swiftvln 0.1.0，支持 Python >= 3.9。训练与评测依赖分开维护，避免 Habitat 0.2.4/Python 3.9 与训练栈混装。

## Conda 环境

本机实际存在的业务环境如下：

| 环境 | 角色 |
| --- | --- |
| swift-vln-train-update | SwiftVLN 主线训练默认环境 |
| swift-vln-eval-update | SwiftVLN 主线评测默认环境 |
| satnav | SatNav 数据与环境工具 |
| streamvln-baseline | StreamVLN baseline |
| navila-baseline | NaVILA baseline |
| openfly-baseline | OpenFly baseline |
| uninavid-baseline | UniNaVid baseline |

旧 swift-vln、swift-vln-base、swift-vln-train、swift-vln-eval 以及 SatVLN pipeline 环境不再使用。

## 默认数据

SwiftVLN/SatNav 当前默认数据版本为 SatNav-v0.1：

- 训练轨迹：/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/trajectory_data
- 评测 episodes：/mnt/data3/jiangjiajun/dataset/satnav_datasets/SatNav-v0.1/episodes/eval/{split}/all_episodes.json
- 场景目录：/mnt/data3/jiangjiajun/dataset/satnav_datasets/scenes
- 默认评测 split：SatNav 为 val_seen；正式比较应同时跑 val_seen 和 val_unseen

训练脚本和 train_queue.sh 已对齐到上述默认路径；自定义数据分别通过 VLN_DATA_PATH 与 EVAL_CONFIG_PATH 覆盖。

## 当前持久化产物

| 路径 | 大小 | 用途 | 保留策略 |
| --- | ---: | --- | --- |
| output/model_zoo/ | 239G | 22 个精选 HF 模型、最终评测结果和必要训练记录 | 模型与历史实验结果的唯一长期副本 |
| output/3DGS-data/ | 94G | 3DGS 场景数据、转换脚本和缓存 | 当前 rebuttal/场景数据 |
| output/PCD-data/ | 42G | PCD 场景与渲染缓存 | 当前 rebuttal/场景数据 |
| output/s2r/ | 405M | S2R 正式 best.pt 与最小训练元数据 | 保留正式权重，不保留 smoke/重复 checkpoint |
| output/rebuttal/ | 292M | episode JSON、GeoTIFF、轨迹预览与坐标说明 | 只保留 episode 及直接相关资料 |

以下工作区级历史目录当前不存在：output/swiftvln、四个 output/*-baseline 原始训练目录、results、logs、swanlog、test_data。新训练或评测仍会按脚本约定重新创建相应运行目录；完成筛选后应把长期资产收口到 model zoo，再清理原始运行目录。

## Model zoo 使用约定

- HF_model/ 是可直接加载的模型副本，包含 config、processor/tokenizer 与独立权重。
- Results/ 是当前最终评测的权威副本；精确指标以各 split 的 evaluation_summary.json 为准。
- Training_Log/ 只在 baseline 与 SwiftVLN 组中保留精选训练元数据，不代表存在 optimizer state。
- 结果 JSON 内的 model_path 是历史 provenance，部分路径已经随原始实验目录清理而失效；重新评测必须使用对应 HF_model 路径。
- output/model_zoo/swiftvln/HF_model 下的主线模型可由 eval_by_name.sh 自动回退找到；backbones 模型应显式传 MODEL_PATH。

完整清单与指标见 [model_zoo.md](model_zoo.md)。

## 当前已知限制

1. 历史 DeepSpeed global_step、latest 和 zero_to_fp32.py 已删除，因此不能恢复当时的 optimizer/scheduler 状态；可做推理、评测或仅权重初始化。
2. S2R 正式 best.pt 完整且独立评测通过，但 train_args.json 引用的 manifest_v2_gta_dedup.jsonl 已不在 runtime/s2r/manifests/；只有 canonical manifest_v1.jsonl 保留。精确复现该次训练前必须重新生成并验证 v2 manifest。
3. model zoo 的历史 provenance 字段不会改写成当前路径，以免伪造来源；当前可执行路径由目录结构和本报告给出。
4. reports/ 不再保存日期快照。需要历史过程时查看 Git，不能把旧路径重新当成当前操作说明。

## 更新触发条件

以下变化发生时必须原位更新本报告：

- 主线脚本、包结构或默认数据版本变化；
- Conda 环境增删或默认环境变化；
- output/ 的长期保留策略变化；
- model zoo、S2R 或 rebuttal 产物发生增删；
- 任何“当前已知限制”被解决或出现新的阻塞项。
