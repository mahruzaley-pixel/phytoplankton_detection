from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image

from ..utils.helpers import load_yaml, setup_logger


class DatasetValidator:
    """Validates YOLO dataset integrity and reports statistics."""

    def __init__(self, data_yaml_path: Union[str, Path], logger=None):
        self.yaml_path = Path(data_yaml_path)
        self.logger = logger or setup_logger("DatasetValidator")

        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found: {self.yaml_path}")

        self.config = load_yaml(self.yaml_path)
        self.base_dir = Path(self.config.get("path", self.yaml_path.parent))
        self.classes = self.config.get("names", {})
        if isinstance(self.classes, list):
            self.classes = {i: name for i, name in enumerate(self.classes)}

    def validate(self) -> Dict[str, Dict[str, int]]:
        """
        Runs validation across train, val, and test splits.
        Returns dictionary of statistics.
        """
        self.logger.info(f"Validating dataset configured at {self.yaml_path}")
        results = {}
        all_class_counts = Counter()

        for split in ["train", "val", "test"]:
            split_rel_path = self.config.get(split)
            if not split_rel_path:
                self.logger.warning(f"Split '{split}' not defined in data.yaml")
                continue

            img_dir = (self.base_dir / split_rel_path).resolve()
            lbl_dir = (self.base_dir / "labels" / split).resolve()

            if not img_dir.exists():
                self.logger.warning(f"Image directory does not exist: {img_dir}")
                continue

            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            corrupt_images = 0
            missing_labels = 0
            empty_labels = 0
            invalid_boxes = 0
            split_boxes = 0
            split_class_counts = Counter()

            for img_path in img_files:
                # 1. Image integrity check
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt_images += 1
                    continue

                # 2. Label check
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    missing_labels += 1
                    continue

                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    empty_labels += 1
                    continue

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        invalid_boxes += 1
                        continue

                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])

                        if cls_id not in self.classes:
                            invalid_boxes += 1
                        elif not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            invalid_boxes += 1
                        else:
                            split_boxes += 1
                            split_class_counts[cls_id] += 1
                            all_class_counts[cls_id] += 1
                    except ValueError:
                        invalid_boxes += 1

            results[split] = {
                "total_images": len(img_files),
                "valid_images": len(img_files) - corrupt_images,
                "corrupt_images": corrupt_images,
                "missing_labels": missing_labels,
                "empty_labels": empty_labels,
                "total_boxes": split_boxes,
                "invalid_boxes": invalid_boxes,
            }

            self.logger.info(
                f"Split '{split}': {len(img_files)} images, {split_boxes} bounding boxes, "
                f"{corrupt_images} corrupt, {invalid_boxes} invalid boxes"
            )

        self.logger.info("Class distribution summary:")
        for cls_id in sorted(self.classes.keys()):
            name = self.classes[cls_id]
            cnt = all_class_counts[cls_id]
            self.logger.info(f"  [{cls_id:02d}] {name:30s}: {cnt:5d} boxes")

        return results
