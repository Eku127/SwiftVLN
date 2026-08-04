"""Shared image resize helpers for model input (448×448) and JPEG encode; 公共图像缩放工具（模型输入 448×448）及 JPEG 编码。"""

from __future__ import annotations

from typing import Any, Tuple

MODEL_INPUT_SIZE: Tuple[int, int] = (448, 448)


def prepare_model_image(frame_bgr: Any, resize_mode: str) -> Any:
    """Resize a BGR frame to 448×448 RGB for model input; 将 BGR 帧缩放为 448×448 RGB 模型输入。

    Supported ``resize_mode``: ``center-crop`` (default in env) or ``stretch``.
    支持的 ``resize_mode``：``center-crop``（环境默认）或 ``stretch``。
    """
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if resize_mode == "stretch":
        resized = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    elif resize_mode == "center-crop":
        height, width = rgb.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        cropped = rgb[top : top + side, left : left + side]
        resized = cv2.resize(cropped, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"unsupported resize mode: {resize_mode}")

    image = Image.fromarray(resized, mode="RGB")
    if image.size != MODEL_INPUT_SIZE or image.mode != "RGB":
        raise RuntimeError(
            f"model input must be 448x448 RGB, got size={image.size}, mode={image.mode}"
        )
    return image


def encode_bgr_as_jpeg(frame_bgr: Any, *, quality: int = 90) -> bytes:
    """Encode a BGR numpy frame as JPEG bytes; 将 BGR numpy 帧编码为 JPEG 字节。"""
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    return encoded.tobytes()
