# Model Zoo 当前清单与指标

最后核验：2026-08-02

## 完整性结论

- 总大小：239G。
- HF 模型：22 个；22/22 均存在 config.json 和可加载权重。
- 评测结果：22 套；22/22 均同时包含 val_seen 与 val_unseen 的 evaluation_summary.json 和 all_results.jsonl。
- 分组：backbones 3 个、baseline 8 个、swiftvln 11 个。
- 历史原始训练/评测目录已经清理，output/model_zoo 是这些模型与结果的当前权威副本。

| 分组 | 模型数 | 结果数 | 大小 | Training_Log |
| --- | ---: | ---: | ---: | --- |
| backbones | 3 | 3 | 37G | 不保留 |
| baseline | 8 | 8 | 119G | 8 个模型目录 |
| swiftvln | 11 | 11 | 84G | 11 个模型目录 |

指标均直接读取对应 Results/<model>/{val_seen,val_unseen}/evaluation_summary.json。SR/SPL 以百分数显示；精确原始值、OS、NE、平均步数和分任务统计请读取 JSON。

## Backbones

| 模型 | Seen SR | Seen SPL | Unseen SR | Unseen SPL |
| --- | ---: | ---: | ---: | ---: |
| swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed | 70.99 | 70.51 | 57.65 | 57.09 |
| swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed | 67.49 | 66.73 | 57.01 | 56.34 |
| swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed | 68.30 | 67.59 | 55.98 | 55.27 |

目录：

- 模型：../output/model_zoo/backbones/HF_model/
- 结果：../output/model_zoo/backbones/Results/
- 汇总工作簿：../output/model_zoo/backbones/Results/backbone_results.xlsx

这三个模型需要显式设置 MODEL_PATH 指向 HF_model 下的具体目录；当前 eval_by_name.sh 不自动搜索 backbones 分组。

## Baselines

| 模型 | Seen SR | Seen SPL | Unseen SR | Unseen SPL |
| --- | ---: | ---: | ---: | ---: |
| navila-satnav-continue-1ep-8f-sample-hk7-fs7-stopx4 | 24.99 | 24.88 | 18.57 | 18.44 |
| navila-satnav-scratch-1ep-8f-sample-hk7-fs7-stopx4 | 18.10 | 18.02 | 13.00 | 12.96 |
| openfly-satnav-continue-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 | 21.10 | 20.99 | 17.12 | 16.91 |
| openfly-satnav-scratch-1ep-actcompact-sample-hk7-fs3-stopx2-stopw0-tail5-stoph1-hist16-lr2e-5 | 13.21 | 13.03 | 11.73 | 11.62 |
| streamvln-satnav-continue-1ep-f32h8s4-lr2e-5 | 70.35 | 69.66 | 58.44 | 57.80 |
| streamvln-satnav-scratch-1ep-f32h8s4-lr2e-5 | 64.30 | 63.66 | 52.25 | 51.78 |
| uninavid-satnav-continue-1ep-lr1e-5 | 49.69 | 49.15 | 36.72 | 36.29 |
| uninavid-satnav-scratch-1ep-lr1e-5 | 25.12 | 24.81 | 20.36 | 20.00 |

目录：

- 模型：../output/model_zoo/baseline/HF_model/
- 结果：../output/model_zoo/baseline/Results/
- 训练记录：../output/model_zoo/baseline/Training_Log/

Baseline 结果格式只要求每个 split 存在 evaluation_summary.json 与 all_results.jsonl。OpenFly 的 val_seen 口径为 4601 条，其余本表模型通常为 val_seen 4574 条、val_unseen 8756 条；横向比较时应注意这一差异。

## SwiftVLN 3B variants

| 模型 | Seen SR | Seen SPL | Unseen SR | Unseen SPL |
| --- | ---: | ---: | ---: | ---: |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-gtc-k512-noembed | 63.53 | 63.20 | 51.35 | 51.04 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-map-g1000-l400-r448-d20-s2-noembed | 59.88 | 59.61 | 48.66 | 48.40 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h0-nomem-pool-s2-noembed | 44.49 | 43.78 | 32.39 | 31.48 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-b2.0-pool-s2-noembed | 66.90 | 66.41 | 55.77 | 55.32 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-initial-noembed | 62.66 | 61.63 | 52.64 | 51.72 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed | 65.87 | 65.50 | 53.71 | 53.27 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-posefilm | 67.77 | 67.50 | 55.42 | 55.17 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-random-pool-s2-noembed | 51.97 | 51.60 | 41.14 | 40.90 |
| swiftvln-satnav-3b-1ep-f32s4-overlap0-sgtc-k512-noembed | 68.26 | 67.85 | 54.47 | 54.12 |
| swiftvln-satnav-3b-1ep-f32s4-overlap16-pf-h8-pool-s2-noembed | 71.32 | 70.87 | 60.34 | 59.88 |
| swiftvln-satnav-3b-1ep-f32s4-overlap4-pf-h8-pool-s2-noembed | 68.71 | 64.01 | 56.77 | 53.73 |

目录：

- 模型：../output/model_zoo/swiftvln/HF_model/
- 结果：../output/model_zoo/swiftvln/Results/
- 训练记录：../output/model_zoo/swiftvln/Training_Log/
- 汇总工作簿：../output/model_zoo/swiftvln/Results/swiftvln_version_results.xlsx

SwiftVLN/backbones 每个 split 还保留 timing_summary.json 与 results_partial.json。主线 eval_by_name.sh 会在普通训练输出不存在时回退到该 HF_model 目录。

## 当前结果解读

- 保留的 3B SwiftVLN 中，overlap16 版本当前 seen/unseen SR 均最高：71.32 / 60.34。
- StreamVLN continue 是当前最强 baseline：seen/unseen SR 为 70.35 / 58.44。
- 3B overlap16 在 unseen SR 上高于 7B backbone（60.34 对 57.65），说明当前结果不能简单按参数规模排序。
- no-memory 版本相对标准 per-frame history 版本明显下降，当前结果支持保留历史记忆机制。
- overlap4 的 SR 与 SPL 差距比其他主线变体更大；使用时应同时看 SPL，不能只按 SR 选型。
- Baseline 的 continue 版本在四个架构上均优于对应 scratch 版本。

## 使用与维护

1. 模型目录名是当前稳定标识；不要再引用旧长实验名或 output/overlapvln 路径。
2. evaluation_summary.json 内的 model_path 仅用于历史 provenance，原始路径可能已删除。
3. 新增或替换模型时，必须保证 HF_model 与 Results 同名，并同时校验两个 split。
4. 更新本报告时从 JSON 重新生成指标，不手工沿用旧 baseline 报告或工作簿中的缓存数值。
