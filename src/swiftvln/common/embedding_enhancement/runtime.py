"""Runtime helpers for attaching embedding enhancement modules to models."""

import torch

from swiftvln.experiment import (
    embedding_uses_pose,
    embedding_uses_uav,
    normalize_embedding_mode,
    pose_fusion_for_embedding_mode,
)


def _emit(logger, message: str) -> None:
    if logger is None:
        print(message)
    elif hasattr(logger, "info"):
        logger.info(message)
    else:
        logger(message)


def _set_alias(model, alias_name: str, value) -> None:
    if hasattr(model, "_modules"):
        model._modules.pop(alias_name, None)
    model.__dict__[alias_name] = value


def _desired_enhancements(embedding_mode: str):
    if embedding_uses_pose(embedding_mode):
        return ["pose"]
    if embedding_uses_uav(embedding_mode):
        return ["uav"]
    return []


def _needs_rebuild_pipeline(model, embedding_mode: str) -> bool:
    if not hasattr(model, "embed_enhance") or model.embed_enhance is None:
        return True
    desired_enhancements = _desired_enhancements(embedding_mode)
    current = list(getattr(model.embed_enhance, "enhancements", {}))
    if current != desired_enhancements:
        return True
    if embedding_uses_pose(embedding_mode):
        pose_module = model.embed_enhance.enhancements["pose"]
        return pose_module.fusion != pose_fusion_for_embedding_mode(embedding_mode)
    return False


def _move_pipeline_to_model_device(model) -> None:
    if not hasattr(model, "embed_enhance") or model.embed_enhance is None:
        return
    if model.embed_enhance.is_empty:
        return

    target_dtype = model.visual.dtype if hasattr(model, "visual") and hasattr(model.visual, "dtype") else None
    try:
        target_device = next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        target_device = getattr(model, "device", torch.device("cpu"))

    to_kwargs = {}
    if target_dtype is not None:
        to_kwargs["dtype"] = target_dtype
    if target_device is not None:
        to_kwargs["device"] = target_device
    if to_kwargs:
        model.embed_enhance = model.embed_enhance.to(**to_kwargs)


def set_embedding_enhancement_aliases(
    model,
    *,
    embedding_mode: str,
) -> None:
    embedding_mode = normalize_embedding_mode(embedding_mode)
    enhancements = getattr(getattr(model, "embed_enhance", None), "enhancements", {})

    if embedding_uses_pose(embedding_mode) and "pose" in enhancements:
        _set_alias(model, "pose_embed", enhancements["pose"])
    else:
        _set_alias(model, "pose_embed", None)

    if embedding_uses_uav(embedding_mode) and "uav" in enhancements:
        _set_alias(model, "uav_adapter", enhancements["uav"])
    else:
        _set_alias(model, "uav_adapter", None)


def configure_embedding_enhancement(
    model,
    *,
    embedding_mode: str,
    uav_adapter_path: str,
    uav_adapter_type: str,
    uav_adapter_apply_scope: str,
    pose_norm_scale: float,
    force_rebuild: bool = False,
    restore_callback=None,
    logger=None,
    log_embed_dim: bool = False,
    log_train_save_note: bool = False,
) -> None:
    embedding_mode = normalize_embedding_mode(embedding_mode)
    if model is None:
        return
    model.config.embedding_mode = embedding_mode

    from swiftvln.common.embedding_enhancement import create_embedding_pipeline

    if force_rebuild or _needs_rebuild_pipeline(model, embedding_mode):
        embed_dim = model.config.hidden_size
        model.embed_enhance = create_embedding_pipeline(
            embed_dim=embed_dim,
            embedding_mode=embedding_mode,
            pose_norm_scale=pose_norm_scale,
            uav_adapter_path=uav_adapter_path,
            uav_adapter_type=uav_adapter_type,
            uav_adapter_apply_scope=uav_adapter_apply_scope,
        )
        if not force_rebuild:
            _emit(logger, f"[SwiftVLN] Rebuilt embed_enhance pipeline in trainer: {model.embed_enhance}")

    _move_pipeline_to_model_device(model)

    if not hasattr(model, "embed_enhance") or model.embed_enhance is None:
        set_embedding_enhancement_aliases(
            model,
            embedding_mode=embedding_mode,
        )
        return

    if not model.embed_enhance.is_empty and restore_callback is not None:
        restore_callback(model)

    if embedding_uses_uav(embedding_mode) and uav_adapter_path:
        resolved_path = model.embed_enhance.enhancements["uav"].load_external_checkpoint(
            uav_adapter_path,
            strict=True,
        )
        _emit(logger, f"[SwiftVLN] Loaded external UAV adapter from: {resolved_path}")

    if not model.embed_enhance.is_empty:
        _emit(logger, f"[SwiftVLN] Embedding enhancement pipeline: {model.embed_enhance}")
        if log_embed_dim:
            _emit(logger, f"  - embed_dim: {model.config.hidden_size}")
        _emit(logger, f"  - Enhancements: {model.embed_enhance.enhancement_names}")
        if log_train_save_note:
            _emit(logger, "  - Module will be trained and saved with checkpoints")

    set_embedding_enhancement_aliases(
        model,
        embedding_mode=embedding_mode,
    )
