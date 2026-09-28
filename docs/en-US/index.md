# SwiftVLN Documentation

SwiftVLN brings trajectory-based training, memory experiments, and online evaluation into one framework. It supports Qwen2.5-VL and Qwen3-VL for SatNav aerial navigation and Habitat indoor navigation.

<p align="center">
  <img src="../assets/readme/swiftvln-framework.png" width="75%" alt="SwiftVLN combines historical memory, a sliding dialogue window, and language instructions to predict navigation actions.">
</p>

## Start here

| Your goal | Guide |
| --- | --- |
| Understand the implementation | [Dual memory and windows](concepts/PIPELINE.md) · [History compression](concepts/MEMORY.md) · [Map memory](concepts/MAP_MEMORY.md) · [Input augmentation](concepts/AUGMENTATION.md) |
| Evaluate a released model | [Installation](getting-started/INSTALLATION.md) · [Checkpoints](getting-started/CHECKPOINTS.md) · [Evaluation](evaluation/README.md) |
| Train a navigation policy | [SatNav data](data/TRAINING_DATA_SATNAV.md) · [Habitat data](data/TRAINING_DATA_HABITAT.md) · [Training](training/README.md) |
| Compare memory designs | [Memory configuration](training/MEMORY.md) |
| Adapt to UAV observations | [SatDronePair](data/SATDRONEPAIR.md) · [Stage-A training](training/S2R_STAGE_A.md) |
| Extend the framework | [Architecture](development/ARCHITECTURE.md) · [Extension guide](development/EXTENDING.md) |

## Training and evaluation workflow

Prepare an environment, download a base model or navigation checkpoint, and configure local data paths. Training uses offline RGB observations and expert actions; evaluation runs the policy online on SatNav or Habitat episodes. Use the same model name to carry memory and window settings from training into evaluation.

The [workflow guide](README.md) connects the setup, data, training, evaluation, and development documents. The reference configuration uses 32-action windows, four actions per prediction, and eight uniformly sampled history frames.

## Project resources

- [Paper](https://arxiv.org/abs/2609.31507): *SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery*.
- [SwiftVLN Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo): reference and memory ablation checkpoints.
- [SatNav Wiki](https://eku127.github.io/SatNav/wiki/): simulator, episodes, satellite scenes, and baseline documentation.
- [Source code](https://github.com/Eku127/SwiftVLN): models, scripts, configuration, and documentation sources.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

Installation <getting-started/INSTALLATION>
Models and checkpoints <getting-started/CHECKPOINTS>
Workflow guide <README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Memory Mechanisms

Dual memory and sliding windows <concepts/PIPELINE>
Historical memory and token compression <concepts/MEMORY>
Map memory <concepts/MAP_MEMORY>
Input augmentation and UAV adaptation <concepts/AUGMENTATION>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Data Preparation

SatNav trajectories <data/TRAINING_DATA_SATNAV>
Habitat trajectories <data/TRAINING_DATA_HABITAT>
Evaluation data <data/EVALUATION_DATA>
SatDronePair <data/SATDRONEPAIR>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Training & Memory

Training <training/README>
Memory configuration <training/MEMORY>
Satellite-to-UAV Stage-A <training/S2R_STAGE_A>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Evaluation

Online evaluation <evaluation/README>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Development

Architecture <development/ARCHITECTURE>
Extending SwiftVLN <development/EXTENDING>
```
