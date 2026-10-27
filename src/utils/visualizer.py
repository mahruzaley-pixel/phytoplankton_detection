from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import colorsys

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image, ImageDraw, ImageFont


def generate_class_colors(num_classes: int = 24) -> List[Tuple[int, int, int]]:
    """Generates visually distinct RGB colors for class bounding boxes."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        sat = 0.85
        val = 0.95
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


class Visualizer:
    """Helper for rendering bounding boxes and labels on images."""

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {}
        num_classes = max(len(self.class_names), 24)
        self.colors = generate_class_colors(num_classes)

    def draw_detections(
        self,
        image: Any,
        boxes: List[List[float]],
        class_ids: List[int],
        scores: Optional[List[float]] = None,
        output_path: Optional[Union[str, Path]] = None,
        box_thickness: int = 3,
    ) -> Image.Image:
        """
        Draws bounding boxes on the image.
        boxes are in [xmin, ymin, xmax, ymax] format.
        """
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif np is not None and isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image if image.shape[2] == 3 else (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cv2 else image))
        elif isinstance(image, Image.Image):
            pil_img = image.copy().convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.load_default()

        for idx, (box, cls_id) in enumerate(zip(boxes, class_ids)):
            color = self.colors[cls_id % len(self.colors)]
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=box_thickness)

            cls_name = self.class_names.get(cls_id, f"Class_{cls_id}")
            label = cls_name
            if scores is not None and idx < len(scores):
                label += f" {scores[idx]:.2f}"

            # Text background
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_bg = [xmin, max(0, ymin - text_h - 4), xmin + text_w + 6, max(0, ymin)]
            draw.rectangle(text_bg, fill=color)
            draw.text((xmin + 3, max(0, ymin - text_h - 2)), label, fill=(255, 255, 255), font=font)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_p)

        return pil_img
