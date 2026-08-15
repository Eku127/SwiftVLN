# SatDronePair Data Generation

This directory contains the repository-only converters that build SwiftVLN
SatDronePair data from DenseUAV, GTA-UAV, SUES-200, and UAV-VisLoc.

Install the optional dependencies and inspect the unified CLI from the
SwiftVLN repository root:

```bash
python -m pip install -e ".[s2r-data]"
python -m tools.s2r.data_generation --help
python -m tools.s2r.data_generation <dataset> <command> --help
```

See the [Chinese SatDronePair guide](../../../docs/zh-CN/data/SATDRONEPAIR.md)
for upstream data sources, conversion commands, output layout, manifest
generation, and quality checks.
