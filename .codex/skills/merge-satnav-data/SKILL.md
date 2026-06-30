---
name: merge-satnav-data
description: Merge two SatNav dataset versions into a new version with preflight overlap analysis, duplicate-safe episode remapping, trajectory_data rewrite, and strict post-merge validation. Use when asked to merge SatNav versions like 0403 plus 0327 into 0404, create a new merged SatNav dataset version, analyze complementarity between two SatNav datasets, or run merge_satnav_data.
---

# Merge SatNav Data

Use this skill when the user wants to merge two SatNav dataset versions into a new `ver_xxxxxx` dataset and needs the result to stay internally consistent.

The canonical repo entrypoint is:

```bash
python3 src/swiftvln/scripts/data_process/merge_satnav_data.py \
    <primary_version> <secondary_version> <output_version>
```

## Workflow

### 1. Preflight Analysis

Always run analyze-only first:

```bash
python3 src/swiftvln/scripts/data_process/merge_satnav_data.py \
    <primary_version> <secondary_version> <output_version> \
    --analyze-only
```

Check these fields in the JSON output:

- `analysis.split_stats.train|val_seen|val_unseen`
- `analysis.remapped_episode_count`
- `analysis.city_stats`

Important rule:

- Do not assume same `scene_id + episode_id` means the two versions are true duplicates.
- If `conflicting_overlap > 0`, the secondary payload differs and must be kept via remapping, not dropped.

### 2. Merge Execution

Default to hardlink mode unless the user explicitly wants full physical copies:

```bash
python3 src/swiftvln/scripts/data_process/merge_satnav_data.py \
    <primary_version> <secondary_version> <output_version> \
    --copy-mode hardlink
```

Behavior:

- Per-city `data/*/VLN_episodes.json` is merged first.
- Identical overlaps are deduplicated.
- Conflicting overlaps are remapped to fresh secondary `episode_id`.
- Merged `episodes/` is regenerated through `process_episodes.py`.
- Secondary `trajectory_data/summary.json`, `annotations.json`, and `images/<scene>_satnav_<id>` are rewritten to the remapped IDs.

### 3. Post-Merge Validation

Inspect these output artifacts:

- `<output_version>/merge_manifest.json`
- `<output_version>/episode_id_remap.jsonl`
- `<output_version>/episodes/train/all_episodes.json`
- `<output_version>/episodes/eval/val_seen/all_episodes.json`
- `<output_version>/episodes/eval/val_unseen/all_episodes.json`
- `<output_version>/trajectory_data/summary.json`
- `<output_version>/trajectory_data/annotations.json`

The script already enforces these checks:

- Source versions exist.
- Output version is empty before merge.
- Merged `scene_id + episode_id` keys are unique.
- Train / eval splits grow unless `--allow-no-growth` is passed.
- `trajectory_data/summary.json` and `annotations.json` cover the same video set.
- `trajectory_data/images/` matches the merged summary video set exactly.
- A remap log is written for every remapped secondary episode.

## Output Conventions

The merge writes:

- `merge_manifest.json`: full merge summary and validation result
- `episode_id_remap.jsonl`: secondary old/new ID mapping
- `data/`: merged city assets and merged `VLN_episodes.json`
- `raw_data/`: merged `VLN_episodes.json`
- `episodes/`: regenerated train / val_seen / val_unseen splits
- `trajectory_data/`: merged images + merged metadata

## Notes

- `--copy-mode hardlink` is preferred for large dataset merges on the same filesystem.
- If the user wants this merged dataset to become the repo default, update:
  - `src/swiftvln/configs/satnav_task.yaml`
  - `src/swiftvln/scripts/train/train_queue.sh`

## Known Example

For `ver_260327 + ver_260403 -> ver_260404`:

- Raw `scene_id + episode_id` overlap exists for all `ver_260403` episodes.
- Those overlaps are content conflicts, not true duplicates.
- The correct merge keeps all `8820` secondary episodes by remapping IDs.
