# OpenFly Baseline Env

- Conda env name: `openfly-baseline`
- Setup script: `baseline/openfly/scripts/setup_env.sh`
- Runtime dependency on SatNav only:
  `pip install -e /mnt/data1/home/jiangjiajun/workspace/SatNav`
- `hf` backend needs a normal HF OpenFly model directory.
- `native` backend additionally needs:
  - a local Prismatic/OpenVLA checkpoint directory such as `baseline/openfly/model/openvlaopenvla-7b-prismatic`
  - a local HF processor/tokenizer source such as `baseline/openfly/model/openfly-agent-7b`
  - access to `meta-llama/Llama-2-7b-hf` weights/tokenizer when first constructing the model
- No AirSim / UnrealCV / ROS2 / TFDS dependencies are required for this baseline.
