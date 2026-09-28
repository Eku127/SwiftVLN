# README Figures

These figures are shared by the English and Chinese repository READMEs. They
come from the manuscript *SatNav: A Scalable Benchmark for Long-Horizon UAV
Vision-Language Navigation from Satellite Imagery* ([paper](https://arxiv.org/abs/2609.31507)).

| Asset | Manuscript source | Content |
| --- | --- | --- |
| `swiftvln-framework.png` | `figures/swiftvln/model-arch-with-window.pdf` | SwiftVLN framework overview from the SwiftVLN section |
| `satellite-uav-pairs.png` | `appendices/figures/swiftvln/sim2real/s2r.pdf` | Paired satellite and UAV imagery from the adapter appendix |

The PNGs were rendered directly from the vector PDFs with Poppler. The figures
retain the original content and labels; the pair counts describe the paper's
adaptation dataset.

To regenerate them, run from the SwiftVLN repository root and set `PAPER_ROOT`
to the manuscript source directory:

```bash
pdftoppm -f 1 -singlefile -scale-to 2400 -png \
  "${PAPER_ROOT}/figures/swiftvln/model-arch-with-window.pdf" \
  docs/assets/readme/swiftvln-framework

pdftoppm -f 1 -singlefile -scale-to 2000 -png \
  "${PAPER_ROOT}/appendices/figures/swiftvln/sim2real/s2r.pdf" \
  docs/assets/readme/satellite-uav-pairs
```
