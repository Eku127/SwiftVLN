# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Image processing utilities for VLN evaluation.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def append_text_to_image(image: np.ndarray, text: str, position: str = 'bottom') -> np.ndarray:
    """
    Append text to image (below or above the image).
    
    Args:
        image: Input image as numpy array (H, W, C)
        text: Text to add
        position: 'bottom' or 'top'
        
    Returns:
        Image with text appended
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    
    # Try to load a font, fallback to default if not available
    font_size = 20
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, ValueError):
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Calculate text size (handle text wrapping for long instructions)
    draw = ImageDraw.Draw(pil_image)
    max_width = pil_image.width - 20  # Leave margins
    
    # Simple text wrapping
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        word_width = bbox[2] - bbox[0]
        
        if current_width + word_width <= max_width or len(current_line) == 0:
            current_line.append(word)
            current_width = word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate total text height
    line_height = font_size + 4
    text_height = len(lines) * line_height
    
    # Create new image with space for text
    padding = 10
    new_height = pil_image.height + text_height + 2 * padding
    new_image = Image.new('RGB', (pil_image.width, new_height), color='white')
    
    if position == 'bottom':
        # Paste original image at top
        new_image.paste(pil_image, (0, 0))
        # Draw text at bottom
        draw = ImageDraw.Draw(new_image)
        y_pos = pil_image.height + padding
        for line in lines:
            draw.text((padding, y_pos), line, fill='black', font=font)
            y_pos += line_height
    else:
        # Draw text at top
        draw = ImageDraw.Draw(new_image)
        y_pos = padding
        for line in lines:
            draw.text((padding, y_pos), line, fill='black', font=font)
            y_pos += line_height
        # Paste original image below text
        new_image.paste(pil_image, (0, text_height + 2 * padding))
    
    return np.array(new_image)
