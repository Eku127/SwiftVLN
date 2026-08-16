# SwiftVLN

English | [简体中文](README_ZH.md)

SwiftVLN is a training and online evaluation framework for vision-and-language navigation research.
It supports aerial navigation with SatNav and indoor navigation with Habitat, covering the complete
workflow from trajectory generation and supervised fine-tuning to Memory experiments and distributed evaluation.

## Features

- Qwen2.5-VL and Qwen3-VL model families;
- one training entry point for SatNav and Habitat offline expert trajectories;
- per-frame, Map, GTC, and Segment-GTC Memory designs;
- history-frame sampling, initial observations, relative poses, and visual embedding enhancements;
- SatNav and Habitat trajectory generation and data validation;
- single- and multi-GPU SatNav evaluation with resume and video visualization;
- SatDronePair data generation and Satellite-to-UAV Stage-A adapter training.

## Getting started

### 1. Installation

SwiftVLN uses separate `swiftvln-train` and `swiftvln-eval` Conda environments. Start by cloning the repository:

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
```

Initialize the third-party submodules required by your training or evaluation workflow, then follow the
[installation guide](docs/en-US/getting-started/INSTALLATION.md) to create the environments.

### 2. Models and data

- [Models and checkpoints](docs/en-US/getting-started/CHECKPOINTS.md)
- [SatNav training data](docs/en-US/data/TRAINING_DATA_SATNAV.md)
- [Habitat training data](docs/en-US/data/TRAINING_DATA_HABITAT.md)
- [SatNav and Habitat evaluation data](docs/en-US/data/EVALUATION_DATA.md)

Store machine-specific model, dataset, and dependency paths in `.local/env.sh`:

```bash
mkdir -p .local
cp local.env.example .local/env.sh
${EDITOR:-vi} .local/env.sh
```

### 3. Training and evaluation

Train on SatNav or Habitat:

```bash
bash scripts/train/train_swiftvln_qwen_vl.sh
```

Evaluate a checkpoint using its full model name:

```bash
bash scripts/eval/eval_by_name.sh "${MODEL_NAME}"
```

See [SwiftVLN training](docs/en-US/training/README.md) and
[SwiftVLN evaluation](docs/en-US/evaluation/README.md) for environment, data, GPU, and Memory configuration.

## Model Zoo

The default SatNav model is the recommended starting point for evaluation or further training:

- [SwiftVLN SatNav 3B reference](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed)

The complete Memory ablation set is available in the
[SwiftVLN SatNav Ablation Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo).

Backbone comparison models:

| Backbone | Model |
| --- | --- |
| Qwen2.5-VL 7B | [SwiftVLN SatNav Qwen2.5-VL 7B](https://huggingface.co/Eku127/swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |
| Qwen3-VL 2B | [SwiftVLN SatNav Qwen3-VL 2B](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |
| Qwen3-VL 8B | [SwiftVLN SatNav Qwen3-VL 8B](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) |

See [Models and checkpoints](docs/en-US/getting-started/CHECKPOINTS.md) for download commands and model-name conventions.

## Documentation

| Topic | Documentation |
| --- | --- |
| Documentation home | [SwiftVLN documentation](docs/README.md) |
| Installation | [Installation](docs/en-US/getting-started/INSTALLATION.md) |
| Model downloads | [Models and checkpoints](docs/en-US/getting-started/CHECKPOINTS.md) |
| Training | [SwiftVLN training](docs/en-US/training/README.md) |
| Memory | [Memory configuration](docs/en-US/training/MEMORY.md) |
| Satellite-to-UAV | [Satellite-to-UAV Stage-A](docs/en-US/training/S2R_STAGE_A.md) |
| Evaluation | [SwiftVLN evaluation](docs/en-US/evaluation/README.md) |
| Architecture | [Architecture](docs/en-US/development/ARCHITECTURE.md) |
| Extending SwiftVLN | [Extending SwiftVLN](docs/en-US/development/EXTENDING.md) |

## Repository layout

```text
SwiftVLN/
├── src/swiftvln/     # Training, models, Memory, Backends, and online evaluation
├── scripts/          # Training, evaluation, and queue entry points
├── tools/s2r/        # SatDronePair and Satellite-to-UAV Stage-A tools
├── environments/     # Training and evaluation Conda environments
├── third_party/      # Pinned external source repositories
└── docs/             # en-US and zh-CN user and development documentation
```

See [Architecture](docs/en-US/development/ARCHITECTURE.md) for core modules and extension boundaries.

## Related projects

- [SatNav](https://github.com/Eku127/SatNav): satellite-image navigation environments, datasets, and evaluation tools.
- [ms-swift](https://github.com/modelscope/ms-swift): the multimodal model training framework used by SwiftVLN.
