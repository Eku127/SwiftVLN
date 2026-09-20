# Editable workflow diagrams

Each diagram has English (`en-US`) and Chinese (`zh-CN`) native draw.io sources
and matching SVG exports. Labels, shapes, and connectors can be edited separately.
The diagrams follow the implementation in `src/swiftvln/`.

| File stem | Purpose | Main code references |
| --- | --- | --- |
| `architecture` | Shared model stack and training/evaluation modules | `training/sft/trainer.py`, `evaluation/runner.py`, `modeling/` |
| `training-flow` | Trajectory, conversation, embeddings, supervision, checkpoint | `training/sft/dataset.py`, `modeling/template.py`, `training/sft/trainer.py` |
| `episode-loop` | Query boundaries and single-action execution | `evaluation/episode_loop.py` |
| `distributed-results` | Global episode sharding, resume, and result aggregation | `evaluation/runner.py`, `evaluation/results.py` |
| `backend-layers` | Environment semantics, factory, and wrapper interface | `backends/specs.py`, `backends/factory.py`, `backends/base.py` |
| `processor-contract` | Matching history processors in training and inference | `modeling/template.py`, `evaluation/inference/encoding.py`, `modeling/history/` |

Export with draw.io Desktop (these exports use 31.4.5):

```bash
drawio --export --format svg --theme light --border 0 --output docs/assets/workflows/architecture.en-US.svg docs/assets/workflows/architecture.en-US.drawio
```

Use `draw.io.exe` on Windows. Keep the white background and native text. Update
both languages, commit source and export together, then follow
[the Wiki build instructions](../../BUILDING.md).

## Compact vector export

Labels use `html=0;whiteSpace=nowrap;` and explicit newlines in the draw.io
source. This produces native SVG text rather than HTML labels with embedded
PNG fallbacks. Keep editable data in the separate `.drawio` file; omit
`--embed-diagram` from SVG exports.

Export and validate all diagrams from the repository root:

```bash
python scripts/export_diagrams.py --drawio /path/to/drawio
```

On Windows, pass the path to `draw.io.exe`. The script rejects raster images,
HTML labels, and embedded diagram source. After changing label text, inspect
both languages for line breaks and overflow, then rebuild the Wiki. Text uses
Arial with the viewer's system fallback for Chinese characters.
