"""Explicit SwiftVLN registration for ms-swift and Transformers."""

from __future__ import annotations


def register_swiftvln_models() -> None:
    """Register templates and both supported Qwen-VL model families.

    Importing :mod:`swiftvln` or :mod:`swiftvln.modeling` has no registration
    side effects. Training and evaluation entrypoints call this function before
    asking ms-swift to parse or load a SwiftVLN model.
    """
    from swift.model import (
        MODEL_MAPPING,
        Model,
        ModelArch,
        ModelGroup,
        ModelMeta,
        register_model,
    )

    from .model import (
        SwiftVLNQwen25VLForConditionalGeneration,
        SwiftVLNQwen25VLLoader,
        SwiftVLNQwen3VLForConditionalGeneration,
        SwiftVLNQwen3VLLoader,
        register_transformers_models,
    )
    from .template import register_swiftvln_templates

    register_swiftvln_templates()
    register_transformers_models()

    if "swiftvln_qwen2_5_vl" not in MODEL_MAPPING:
        register_model(
            ModelMeta(
                model_type="swiftvln_qwen2_5_vl",
                model_groups=[
                    ModelGroup(
                        [
                            Model(
                                "swiftvln-qwen2.5-vl-3b",
                                "Qwen/Qwen2.5-VL-3B-Instruct",
                            ),
                            Model(
                                "swiftvln-qwen2.5-vl-7b",
                                "Qwen/Qwen2.5-VL-7B-Instruct",
                            ),
                        ]
                    )
                ],
                template="swiftvln_qwen2_5_vl",
                model_arch=ModelArch.qwen2_vl,
                architectures=[
                    SwiftVLNQwen25VLForConditionalGeneration.__name__
                ],
                requires=["transformers>=4.49", "qwen_vl_utils>=0.0.6"],
                tags=["vision", "vln", "navigation", "compression"],
                is_multimodal=True,
                loader=SwiftVLNQwen25VLLoader,
            )
        )

    if "swiftvln_qwen3_vl" not in MODEL_MAPPING:
        register_model(
            ModelMeta(
                model_type="swiftvln_qwen3_vl",
                model_groups=[
                    ModelGroup(
                        [
                            Model("swiftvln-qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct"),
                            Model("swiftvln-qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct"),
                            Model("swiftvln-qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct"),
                        ]
                    )
                ],
                template="swiftvln_qwen3_vl",
                model_arch=ModelArch.qwen3_vl,
                architectures=[SwiftVLNQwen3VLForConditionalGeneration.__name__],
                requires=["transformers>=4.57", "qwen_vl_utils>=0.0.14", "decord"],
                tags=["vision", "vln", "navigation", "compression", "qwen3"],
                is_multimodal=True,
                loader=SwiftVLNQwen3VLLoader,
            )
        )
