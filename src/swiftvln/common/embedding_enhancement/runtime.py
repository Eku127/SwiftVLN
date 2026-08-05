"""Runtime helpers for attaching embedding enhancement modules to models."""

import torch

from swiftvln.experiment import embedding_from_flags


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


def _desired_enhancements(use_pose_embed: bool, use_uav_adapter: bool):
    desired = []
    if use_pose_embed:
        desired.append("pose")
    if use_uav_adapter:
        desired.append("uav")
    return desired


def _needs_rebuild_pipeline(model, desired_enhancements) -> bool:
    if not hasattr(model, "embed_enhance") or model.embed_enhance is None:
        return True
    if len(desired_enhancements) == 0:
        return False
    if model.embed_enhance.is_empty:
        return True
    return any(
        name not in getattr(model.embed_enhance, "enhancements", {})
        for name in desired_enhancements
    )


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
    use_pose_embed: bool,
    use_uav_adapter: bool,
    clear_disabled_aliases: bool = True,
) -> None:
    enhancements = getattr(getattr(model, "embed_enhance", None), "enhancements", {})

    if use_pose_embed and "pose" in enhancements:
        _set_alias(model, "pose_embed", enhancements["pose"])
    elif clear_disabled_aliases or not hasattr(model, "pose_embed"):
        _set_alias(model, "pose_embed", None)

    if use_uav_adapter and "uav" in enhancements:
        _set_alias(model, "uav_adapter", enhancements["uav"])
    elif clear_disabled_aliases or not hasattr(model, "uav_adapter"):
        _set_alias(model, "uav_adapter", None)


def configure_embedding_enhancement(
    model,
    *,
    use_pose_embed: bool,
    use_uav_adapter: bool,
    uav_adapter_path: str,
    uav_adapter_type: str,
    uav_adapter_apply_scope: str,
    pose_fusion_method: str,
    pose_norm_scale: float,
    force_rebuild: bool = False,
    restore_callback=None,
    clear_disabled_aliases: bool = True,
    logger=None,
    log_embed_dim: bool = False,
    log_train_save_note: bool = False,
) -> None:
    embedding_from_flags(
        use_pose_embed,
        use_uav_adapter,
        pose_fusion_method,
    )
    if model is None:
        return

    from swiftvln.common.embedding_enhancement import create_embedding_pipeline

    desired_enhancements = _desired_enhancements(use_pose_embed, use_uav_adapter)
    if force_rebuild or _needs_rebuild_pipeline(model, desired_enhancements):
        embed_dim = model.config.hidden_size
        model.embed_enhance = create_embedding_pipeline(
            embed_dim=embed_dim,
            use_pose_embed=use_pose_embed,
            use_uav_adapter=use_uav_adapter,
            pose_fusion=pose_fusion_method,
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
            use_pose_embed=use_pose_embed,
            use_uav_adapter=use_uav_adapter,
            clear_disabled_aliases=clear_disabled_aliases,
        )
        return

    if not model.embed_enhance.is_empty and restore_callback is not None:
        restore_callback(model)

    if use_uav_adapter and uav_adapter_path and "uav" in model.embed_enhance.enhancements:
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
        use_pose_embed=use_pose_embed,
        use_uav_adapter=use_uav_adapter,
        clear_disabled_aliases=clear_disabled_aliases,
    )
