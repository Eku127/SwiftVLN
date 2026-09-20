# Implementation-guide assets

## Paper map examples

The four PNGs are copied unchanged from the SatNav paper source:

- `appendices/figures/swiftvln/map_memory/London-2_ann91376_global.png`
- `appendices/figures/swiftvln/map_memory/London-2_ann91376_local.png`
- `appendices/figures/swiftvln/map_memory/NewYork-1_ann99962_global.png`
- `appendices/figures/swiftvln/map_memory/NewYork-1_ann99962_local.png`

They appear in **Memory Design Details → Map Memory**, in the figure
**Examples of map memory**, for `London-2_Boundary_ID-1279` and
`NewYork-1_Route_ID-999`. Each pair shows the global map followed by the local map. Blue marks the start,
yellow marks position and heading, and red marks the trajectory.

Both language versions share these original assets and set display width in
their pages. The diagrams describe the paper example; current map rendering
defaults are documented alongside it in the implementation guide.

## Editable implementation diagrams

The `diagrams/` directory contains native draw.io sources and their SVG exports.
Each diagram has English (`en-US`) and Chinese (`zh-CN`) versions. Shapes,
labels, and connectors remain individually editable in draw.io.

| File stem | Content |
| --- | --- |
| `dual-memory` | Historical and recent context, prompt assembly, and action execution |
| `sliding-window` | A 32-step window with two retained query turns |
| `memory-comparison` | Per-frame pooling, GTC, and STC input/output organization |
| `map-construction` | Footprints, masks, map crops, and the default 128-token memory |
| `input-enhancement` | Pose FiLM and UAV adapter paths before history compression |

Open a `.drawio` file in draw.io, edit it, and export the corresponding `.svg`.
The Wiki embeds SVGs and links their editable sources. Export with a white
background and light theme; keep the canvas dimensions and native text.
The exports were produced with draw.io Desktop 31.4.5.

Example command from the repository root, with the draw.io executable on PATH:

```bash
drawio --export --format svg --theme light --border 0 --output docs/assets/concepts/diagrams/map-construction.en-US.svg docs/assets/concepts/diagrams/map-construction.en-US.drawio
```

On Windows, use `draw.io.exe` in place of `drawio` (or its full path). Export
both languages after changing a diagram. Commit the `.drawio` and `.svg` files
together, then rebuild the documentation as described in `docs/BUILDING.md`.

The diagrams explain the current implementation. Values in examples, such as
the Qwen2.5-VL 448 × 448 token grid and window-overlap settings, are labeled on
the canvas. The map pipeline uses compression stride 2 and `adaptive_start`.

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
