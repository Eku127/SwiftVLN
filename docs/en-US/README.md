# SwiftVLN Documentation

[简体中文](../zh-CN/README.md) | English

The SwiftVLN documentation is organized by research workflow. New users should begin with installation, model checkpoints, and data preparation. If your environment and data are already available, proceed directly to training or evaluation.

For SatNav scenes, [request prepared GeoTIFFs](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1) or [generate them with your own API credentials](data/TRAINING_DATA_SATNAV.md).

## Implementation principles

- [Dual memory and sliding windows](concepts/PIPELINE.md): observations, action chunks, window updates, and supervision.
- [Historical memory and token compression](concepts/MEMORY.md): sampling, pooling, GridToMe, GTC, and STC computations.
- [Map memory](concepts/MAP_MEMORY.md): explored masks, global/local views, and token budgets.
- [Input augmentation and UAV adaptation](concepts/AUGMENTATION.md): initial views, pose FiLM, and Stage-A alignment objectives.

## 1. Installation and setup

1. [Installation](getting-started/INSTALLATION.md): create the training and evaluation environments.
2. [Models and checkpoints](getting-started/CHECKPOINTS.md): download Qwen base models and released SwiftVLN checkpoints.
3. [SatNav training data](data/TRAINING_DATA_SATNAV.md): prepare episodes, GeoTIFF scenes, and offline trajectories.
4. [Habitat training data](data/TRAINING_DATA_HABITAT.md): prepare R2R, RxR, and EnvDrop trajectories.
5. [Evaluation data](data/EVALUATION_DATA.md): prepare SatNav and Habitat resources for online evaluation.

## 2. Training

- [SwiftVLN training](training/README.md): shared training configuration, SatNav training, Habitat training, and resume.
- [Memory configuration](training/MEMORY.md): history-frame sampling, input augmentation, and long-term Memory compression.
- [Satellite-to-UAV Stage-A](training/S2R_STAGE_A.md): SatDronePair manifests, adapter training, and retrieval evaluation.
- [SatDronePair generation](data/SATDRONEPAIR.md): convert UAV–satellite datasets for Satellite-to-UAV Stage-A.
- [SatNav training data](data/TRAINING_DATA_SATNAV.md): generate and validate SatNav offline trajectories.
- [Habitat training data](data/TRAINING_DATA_HABITAT.md): generate and validate R2R, RxR, and EnvDrop trajectories.

## 3. Evaluation

- [Models and checkpoints](getting-started/CHECKPOINTS.md): download the default SatNav model, ablation models, and backbone models.
- [Evaluation data](data/EVALUATION_DATA.md): prepare SatNav episodes and GeoTIFF scenes or Habitat R2R and MP3D data.
- [SwiftVLN evaluation](evaluation/README.md): configure and run single- or multi-GPU online evaluation.

## 4. Development

- [Architecture](development/ARCHITECTURE.md): training and evaluation pipelines, module boundaries, and dependency direction.
- [Extending SwiftVLN](development/EXTENDING.md): add an environment backend, history processor, embedding enhancement, or model family.

## 5. Find documentation by task

| Task | Documentation |
| --- | --- |
| Install from source | [Installation](getting-started/INSTALLATION.md) |
| Download a base model or checkpoint | [Models and checkpoints](getting-started/CHECKPOINTS.md) |
| Prepare SatNav training trajectories | [SatNav training data](data/TRAINING_DATA_SATNAV.md) |
| Prepare Habitat training trajectories | [Habitat training data](data/TRAINING_DATA_HABITAT.md) |
| Prepare SatNav or Habitat evaluation data | [Evaluation data](data/EVALUATION_DATA.md) |
| Train SwiftVLN | [SwiftVLN training](training/README.md) |
| Configure a Memory experiment | [Memory configuration](training/MEMORY.md) |
| Generate SatDronePair data | [SatDronePair generation](data/SATDRONEPAIR.md) |
| Train a Satellite-to-UAV adapter | [Satellite-to-UAV Stage-A](training/S2R_STAGE_A.md) |
| Evaluate a checkpoint | [SwiftVLN evaluation](evaluation/README.md) |
| Modify core modules | [Architecture](development/ARCHITECTURE.md) |

## 6. Conventions

- Run commands from the SwiftVLN repository root unless stated otherwise.
- Replace `/path/to/...` with the corresponding path on your machine.
- Keep machine-specific paths, models, datasets, and credentials in local configuration.
- Smoke tests validate execution paths and are not performance benchmarks.
- Each procedure provides an expected output or a condition that can be checked after completion.
