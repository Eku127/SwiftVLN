# SatNav: Scaling Up Long-Horizon UAV Visual Language Navigation via Satellite Imagery

## Positioning

This paper is primarily a **dataset/benchmark** paper with a systems-oriented scope:
- a new task setting (top-down UAV VLN),
- a scalable data construction pipeline,
- a unified evaluation platform,
- strong baselines and analysis for long-horizon navigation and memory.

## Core Story in One Sentence

We introduce the first large-scale benchmark for top-down UAV visual-language navigation, built from satellite-map crops with an automated pipeline, and show that this setting exposes long-horizon memory challenges while enabling transfer to real aerial navigation.

## Introduction Logic

1. **Why this matters (application + gap):**
   City-scale UAV missions require language-guided navigation over long trajectories, but existing VLN benchmarks do not match this requirement well.

2. **Gap 1: Perspective inertia from indoor VLN:**
   Existing UAV-VLN setups often inherit the egocentric/front-view convention (often with depth) from indoor VLN, while aerial/top-down view is also a natural and important option in UAV navigation but remains under-explored.

3. **Gap 2: Scalability bottleneck in data construction:**
   Existing pipelines depend on 3D assets and heavy rendering/simulation; this is expensive and difficult to scale to diverse city-level long-horizon scenarios.

4. **Gap 3: Limited stress-testing of memory and loop reasoning:**
   Many episodes are short and mostly linear, which under-tests long-range memory, loop closure, and counting under instruction grounding.

5. **Our key idea and system:**
   Use satellite-map crops as top-down visual observations to build a scalable benchmark and platform, with three task families (Boundary, LandmarkSet, Road) that explicitly stress memory, orientation, and counting.

6. **What we find and why it is important:**
   Representative methods (SatNav-Baseline, StreamVLN, NaviLA, UniNaVid) show clear performance gaps in this setting; staged training from satellite data to aerial/real data supports transfer to real UAV deployment.

## Introduction Draft

Visual-language navigation (VLN) has made strong progress in indoor and simulator-centric settings, but real-world UAV navigation poses a different systems challenge: language-guided decision-making at city scale, over long trajectories, with strong memory demands. Existing benchmarks only partially cover this requirement and leave a gap between current evaluation protocols and practical UAV usage.

A key issue is perspective inertia. Existing UAV-VLN settings often follow indoor VLN conventions and continue to use egocentric front-view observations, frequently with depth signals. However, UAV systems also have access to aerial/top-down visual information as a natural alternative. This perspective provides rich geometric and semantic structure for navigation, yet remains under-explored in mainstream VLN benchmarks.

A second bottleneck is scalability. Many VLN datasets rely on 3D assets and rendering pipelines, which are costly to build and maintain at city scale. This dependence limits geographic diversity, trajectory length, and iteration speed. The problem is especially critical for outdoor UAV scenarios, where high-quality large-scale 3D assets are difficult to obtain and simulation throughput can become a practical constraint.

A third limitation is evaluation coverage for long-horizon memory. Existing episodes are often short and largely linear. Such settings are insufficient for testing loop closure, long-range dependency tracking, and counting-oriented instruction grounding, all of which are central for real city traversal and patrol-like UAV tasks.

To address these limitations, we present **SatNav-Bench** (name placeholder), the first large-scale benchmark for top-down UAV VLN. The core idea is to use satellite-map crops as a scalable proxy for UAV top-down observations, enabling efficient data generation without heavy 3D rendering. We build an automated pipeline with minimal human intervention and design three complementary task families: **Boundary** (loop closure and memory), **LandmarkSet** (orientation and spatial grounding), and **Road** (road following and counting).

The benchmark contains approximately **1,000,000?** training episodes and **100,000?** evaluation episodes, with an average route length of about **800m?** (numbers to be finalized). We further provide a unified platform for reproducible evaluation and compare representative methods, including **SatNav-Baseline**, **StreamVLN**, **NaviLA**, and **UniNaVid**. Results show that current methods struggle under top-down long-horizon settings, especially on memory-intensive tasks such as Boundary and Road.

Finally, we validate a practical transfer path from large-scale satellite pretraining to aerial/real data adaptation, and demonstrate deployment feasibility on real UAV top-down imagery. By releasing the benchmark, platform, and data-generation pipeline, we aim to establish a systems-ready testbed for long-horizon UAV VLN research.

## Contributions

1. We present the **first** large-scale benchmark for **top-down UAV visual-language navigation**, explicitly targeting long-horizon city-scale settings.
2. We propose a scalable and automated data construction pipeline that uses satellite-map crops instead of heavy 3D rendering, substantially improving dataset scalability.
3. We introduce three task families (Boundary, LandmarkSet, Road) and a unified evaluation platform that systematically probes memory, orientation grounding, and counting in UAV VLN.
4. We benchmark representative methods (SatNav-Baseline, StreamVLN, NaviLA, UniNaVid), showing substantial headroom in top-down long-horizon VLN, and validate transfer from satellite-trained models to real UAV deployment.
5. We will open-source the benchmark, platform, and full data-generation pipeline to support reproducible and extensible research.

## SatNav-Baseline: Optional Configurable Components (from OverlapVLN)

| Option (flag) | Choices (default) | Applies to | What it controls / expected effect |
|---|---|---|---|
| `num_overlap` | integer (`16`) | all settings | Overlap size between adjacent sliding windows. Larger overlap improves cross-window continuity but increases repeated tokens and training cost. |
| `history_processor_type` | `per_frame` / `gtc` / `segment_gtc` (`per_frame`) | all settings | Memory construction strategy. `per_frame` keeps frame-wise structure; `gtc` removes cross-frame redundancy; `segment_gtc` keeps coarse temporal order while clustering. |
| `log_base` | float (`1.0`) | `per_frame` | History sampling distribution. `1.0` is uniform; `>1.0` biases toward recent frames (log-like sampling), usually helping recency-sensitive actions. |
| `compress_stride` | integer (`2`) | `per_frame` | Spatial compression ratio for each history frame (e.g., stride 2 gives about 4x token reduction per frame). Larger stride is cheaper but may lose small-object details. |
| `use_tome` | `true/false` (`false`) | `per_frame` | Switch per-frame compression from average pooling to GridToMe token merging. Often preserves semantics better (especially small structures) at a small speed cost. |
| `use_pixel_embed` | `true/false` (`false`) | all settings | Adds learnable pixel-coordinate embedding enhancement to visual tokens, improving top-down spatial grounding. |
| `use_pose_embed` | `true/false` (`false`) | all settings | Adds per-image pose embedding (`delta_forward`, `delta_right`, heading terms), helping motion-consistent reasoning across frames. |
| `pose_fusion_method` | `additive` / `film` (`additive`) | when `use_pose_embed=true` | Controls how pose signals are fused into visual embeddings; `film` is more expressive but can be less stable than additive fusion. |
| `system_prompt_setting` | `vanilla` / `initial` (`vanilla`) | all settings | Prompt strategy. `initial` adds the first-frame observation to the system prompt (extra visual context) with additional token overhead. |
