from __future__ import annotations

from swiftvln.deployment.model_resolver import OverlapVLNDeploySpec
from swiftvln.deployment.policy import OverlapVLNBaselinePolicy


def load_overlapvln_policy(spec: OverlapVLNDeploySpec) -> OverlapVLNBaselinePolicy:
    import swiftvln.model  # noqa: F401
    from swift.llm import get_model_tokenizer
    import torch

    model, processor = get_model_tokenizer(
        model_id_or_path=spec.checkpoint_path,
        model_type="overlapvln_qwen2_5_vl",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_impl="flash_attn",
        use_pixel_embed=False,
        use_pose_embed=False,
        use_uav_adapter=False,
    )
    model.eval()
    return OverlapVLNBaselinePolicy(model=model, processor=processor, spec=spec)
