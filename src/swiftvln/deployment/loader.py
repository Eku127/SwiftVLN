from __future__ import annotations

from swiftvln.deployment.model_resolver import SwiftVLNDeploySpec
from swiftvln.deployment.policy import SwiftVLNBaselinePolicy


def load_swiftvln_policy(spec: SwiftVLNDeploySpec) -> SwiftVLNBaselinePolicy:
    import swiftvln.model  # noqa: F401
    from swift.model import get_model_processor
    import torch

    model, processor = get_model_processor(
        model_id_or_path=spec.checkpoint_path,
        model_type="swiftvln_qwen2_5_vl",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_impl="flash_attn",
        use_pose_embed=False,
        use_uav_adapter=False,
    )
    model.eval()
    return SwiftVLNBaselinePolicy(model=model, processor=processor, spec=spec)
