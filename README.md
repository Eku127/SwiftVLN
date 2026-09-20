<h1 align="center">SwiftVLN</h1>

<p align="center">
  <strong>
    SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery
  </strong>
</p>

<p align="center">
  <a href="https://eku127.github.io/">Jiajun Jiang</a><sup>1,*</sup> &nbsp; <a href="mailto:chua183@connect.hkust-gz.edu.cn">Chunliang Hua</a><sup>1,*</sup> &nbsp;
  <a href="mailto:chenzichun@idea.edu.cn">Zichun Chen</a><sup>2</sup> &nbsp; <a href="mailto:wuyanxing@idea.edu.cn">Yanxing Wu</a><sup>2</sup><br>
  <a href="mailto:yangzeyuan@idea.edu.cn">Zeyuan Yang</a><sup>2</sup> &nbsp; <a href="https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page/SONG-Jie/jsongroas">Jie Song</a><sup>1,3</sup> &nbsp;
  <a href="https://github.com/xiahaa">Xiao Hu</a><sup>1,2,†</sup>
</p>

<p align="center">
  <sup>1</sup> HKUST(GZ) &nbsp;&nbsp;
  <sup>2</sup> LASER, IDEA &nbsp;&nbsp;
  <sup>3</sup> HKUST
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=hOEniyN6hl"><img src="https://img.shields.io/badge/Paper-OpenReview-B31B1B" alt="SatNav paper on OpenReview"></a>
  <a href="docs/en-US/README.md"><img src="https://img.shields.io/badge/Docs-English-2878D0" alt="English documentation"></a>
  <a href="https://github.com/Eku127/SatNav"><img src="https://img.shields.io/badge/Benchmark-SatNav-43874A" alt="SatNav benchmark"></a>
  <a href="https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo"><img src="https://img.shields.io/badge/Models-Hugging%20Face-FFD21E" alt="SwiftVLN Model Zoo on Hugging Face"></a>
</p>

<p align="center">
  English &nbsp;|&nbsp; <a href="README_ZH.md">简体中文</a>
</p>

<p align="center">
  <a href="docs/assets/readme/swiftvln-framework.png">
    <img src="docs/assets/readme/swiftvln-framework.png" width="75%" alt="SwiftVLN architecture: historical observations pass through a memory module, recent observations form a sliding dialogue window, and a vision-language model combines both with the instruction to predict navigation actions.">
  </a>
</p>

<p align="center">
  <em>Combine a sliding dialogue window with interchangeable long-term memory to follow extended navigation instructions.</em>
</p>

SwiftVLN brings trajectory-based training, memory experiments, and online evaluation into one framework. Built on **ms-swift**, it supports **Qwen2.5-VL and Qwen3-VL** for aerial navigation with **SatNav** and indoor navigation with **Habitat**.

Introduced in [*SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery*](https://openreview.net/forum?id=hOEniyN6hl), SwiftVLN provides a shared training and evaluation pipeline for studying how visual history, spatial cues, and memory compression affect navigation.

## Quick Start

**Evaluate the released SwiftVLN 3B reference model on a SatNav episode.** The workflow uses a CUDA GPU, the `swiftvln-eval` environment, a trained checkpoint, evaluation episodes, and the corresponding GeoTIFF scenes. Commands below run in Bash from the repository root.

Clone the repository and initialize the SatNav evaluation dependencies:

```bash
git clone https://github.com/Eku127/SwiftVLN.git
cd SwiftVLN
export SWIFTVLN_ROOT="${PWD}"
git submodule update --init third_party/ms-swift third_party/SatNav

mkdir -p .local
cp local.env.example .local/env.sh
```

Follow the [installation guide](docs/en-US/getting-started/INSTALLATION.md) to create the environment. Prepare the [evaluation data](docs/en-US/data/EVALUATION_DATA.md), then fill in the Conda, SatNav source, episode, and scene paths in `.local/env.sh`.

Activate the environment and download the reference checkpoint:

```bash
source .local/env.sh
source "${SWIFTVLN_CONDA_SH}"
conda activate swiftvln-eval
python -m pip install --upgrade huggingface_hub

export MODEL_NAME=swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed
export MODEL_PATH="${SWIFTVLN_ROOT}/output/model_zoo/swiftvln/HF_model/${MODEL_NAME}"
hf download "Eku127/${MODEL_NAME}" --local-dir "${MODEL_PATH}"
```

Run one episode on the first GPU:

```bash
EVAL_SPLIT=val_seen \
CUDA_DEVICES=0 \
MAX_EPISODES=1 \
bash scripts/eval/eval_by_name.sh "${MODEL_NAME}"
```

The evaluator runs an online rollout and writes per-episode results and `evaluation_summary.json`. Continue with the [evaluation guide](docs/en-US/evaluation/README.md) for full splits, multiple GPUs, resume, and video output.

To train your own model, prepare the `swiftvln-train` environment and offline expert trajectories using the [training guide](docs/en-US/training/README.md). SatNav and Habitat share the `scripts/train/train_swiftvln_qwen_vl.sh` entry point.

## Memory for Long-Horizon Navigation

SwiftVLN keeps recent image-action turns in a **sliding dialogue window** and represents earlier observations with **long-term memory**. Window overlap preserves recent context when the dialogue advances; memory modules control which older information remains available to the model.

| Component | What you can compare |
| --- | --- |
| Dialogue window | Window length, action prediction horizon, and overlap between consecutive windows |
| History sampling | Uniform, random, or recency-biased selection of past observations |
| Per-frame memory | Spatial pooling or GridToMe compression of individual history frames |
| Token clustering | Global Token Clustering (GTC) or temporal Segment-GTC (`sgtc`; STC in the paper) |
| Map memory | Global and local map representations for SatNav |
| Input augmentation | Initial-frame prompting, relative pose cues, and visual embedding enhancements |

The reference configuration uses **32-action windows**, **4 actions per prediction**, and **8 uniformly sampled history frames** with per-frame pooling. See [Memory configuration](docs/en-US/training/MEMORY.md) for experiment commands and [Architecture](docs/en-US/development/ARCHITECTURE.md) for the implementation.

## Dataset & Model Zoo

Use [SatNav-Episodes-v0.1](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1) for satellite-image navigation, or prepare Habitat trajectories for indoor navigation. Training consumes offline RGB observations and expert actions; online evaluation uses episodes, scenes, and a model checkpoint.

| Resource | Where to start |
| --- | --- |
| SatNav episodes and splits | [Download on Hugging Face](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1) |
| SatNav scenes and training trajectories | [SatNav data preparation](docs/en-US/data/TRAINING_DATA_SATNAV.md) |
| Habitat R2R, RxR, and EnvDrop training trajectories | [Habitat data preparation](docs/en-US/data/TRAINING_DATA_HABITAT.md) |
| SatNav and Habitat evaluation resources | [Evaluation data](docs/en-US/data/EVALUATION_DATA.md) |
| Memory ablation checkpoints | [SwiftVLN SatNav Ablation Model Zoo](https://huggingface.co/collections/Eku127/swiftvln-satnav-ablation-model-zoo) |

Released SatNav models with the reference memory configuration:

| Backbone | Checkpoint | Role |
| --- | --- | --- |
| Qwen2.5-VL 3B | [Download](https://huggingface.co/Eku127/swiftvln-satnav-3b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Default reference model |
| Qwen2.5-VL 7B | [Download](https://huggingface.co/Eku127/swiftvln-satnav-7b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Backbone comparison |
| Qwen3-VL 2B | [Download](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-2b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Backbone comparison |
| Qwen3-VL 8B | [Download](https://huggingface.co/Eku127/swiftvln-satnav-qwen3vl-8b-1ep-f32s4-overlap0-pf-h8-pool-s2-noembed) | Backbone comparison |

See [Models and checkpoints](docs/en-US/getting-started/CHECKPOINTS.md) for downloads, the complete ablation list, and model-name conventions.

## Satellite-to-UAV Adaptation

The Satellite-to-UAV adapter maps UAV visual tokens into the satellite feature space of a frozen vision encoder. Training uses paired satellite and UAV images with contrastive and cosine-alignment objectives. At inference, the adapter connects UAV observations to the SatNav-trained navigation model.

<p align="center">
  <a href="docs/assets/readme/satellite-uav-pairs.png">
    <img src="docs/assets/readme/satellite-uav-pairs.png" width="75%" alt="The paper's 24,467 paired satellite and UAV images: data distribution across GTA-UAV, UAV-VisLoc, DenseUAV, and SUES, with example pairs from each source.">
  </a>
</p>

<p align="center">
  <em>Paired satellite and UAV imagery used in the paper's adaptation experiments.</em>
</p>

Prepare the four source datasets with [SatDronePair generation](docs/en-US/data/SATDRONEPAIR.md), then follow [Satellite-to-UAV Stage-A training](docs/en-US/training/S2R_STAGE_A.md) to train and evaluate the adapter.

## Documentation

| I want to… | Guide |
| --- | --- |
| Install SwiftVLN and load a checkpoint | [Installation](docs/en-US/getting-started/INSTALLATION.md) · [Models and checkpoints](docs/en-US/getting-started/CHECKPOINTS.md) |
| Prepare training or evaluation data | [SatNav](docs/en-US/data/TRAINING_DATA_SATNAV.md) · [Habitat](docs/en-US/data/TRAINING_DATA_HABITAT.md) · [Evaluation data](docs/en-US/data/EVALUATION_DATA.md) |
| Train or evaluate a navigation model | [Training](docs/en-US/training/README.md) · [Evaluation](docs/en-US/evaluation/README.md) |
| Compare memory designs | [Memory configuration](docs/en-US/training/MEMORY.md) |
| Adapt satellite features to UAV observations | [SatDronePair](docs/en-US/data/SATDRONEPAIR.md) · [Stage-A training](docs/en-US/training/S2R_STAGE_A.md) |
| Add a model, memory module, or environment | [Architecture](docs/en-US/development/ARCHITECTURE.md) · [Extending SwiftVLN](docs/en-US/development/EXTENDING.md) |

Browse the complete documentation in [English](docs/en-US/README.md) or [简体中文](docs/zh-CN/README.md).

<details>
<summary><strong>Repository structure</strong></summary>

```text
src/swiftvln/  Models, memory, training, environment backends, and online evaluation
scripts/      Training, evaluation, and task queue entry points
tools/s2r/    SatDronePair preparation and Satellite-to-UAV tools
environments/ Separate training and evaluation Conda environments
third_party/  Pinned ms-swift, SatNav, and Habitat-Lab sources
docs/         English and Chinese documentation, with shared figures
```

</details>

## Acknowledgements

SwiftVLN builds on [ms-swift](https://github.com/modelscope/ms-swift) and the [Qwen-VL](https://github.com/QwenLM/Qwen3-VL) model family. Its training and evaluation design draws on [StreamVLN](https://github.com/InternRobotics/StreamVLN), including the use of short-term dialogue context and long-term visual memory.

We thank the [SatNav](https://github.com/Eku127/SatNav) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) projects for their navigation environments, and the DenseUAV, GTA-UAV, SUES-200, and UAV-VisLoc teams for the datasets used in Satellite-to-UAV adaptation. Dataset sources and preparation steps are listed in the [SatDronePair guide](docs/en-US/data/SATDRONEPAIR.md).
