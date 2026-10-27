import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from ..utils.helpers import load_yaml, save_yaml, setup_logger


class VOCtoYOLOConverter:
    """
    Converts Pascal VOC formatted phytoplankton dataset into YOLO format.
    Supports full conversion and lightweight mini conversion for local testing.
    """

    def __init__(
        self,
        raw_dataset_dir: Union[str, Path],
        output_dir: Union[str, Path],
        classes_config_path: Optional[Union[str, Path]] = None,
        logger=None,
    ):
        self.raw_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)
        self.logger = logger or setup_logger("VOCtoYOLOConverter")

        # Directories in VOC
        self.annot_dir = self.raw_dir / "Annotations"
        self.images_dir = self.raw_dir / "JPEGImages"
        self.imagesets_dir = self.raw_dir / "ImageSets" / "Main"

        # Load class mappings
        if classes_config_path and Path(classes_config_path).is_file():
            config = load_yaml(classes_config_path)
            self.class_names = config.get("names", {})
            if isinstance(self.class_names, list):
                self.class_names = {i: name for i, name in enumerate(self.class_names)}
        else:
            # Default canonical 24 classes
            self.class_names = {
                0: "bacteriastrum",
                1: "biddulphia",
                2: "ceratium carriense",
                3: "ceratium furca",
                4: "ceratium fusus",
                5: "ceratium trichoceros",
                6: "ceratium tripos",
                7: "chaetoceros",
                8: "corethron",
                9: "coscinodiscus",
                10: "coscinodiscus flank",
                11: "detonula pumila",
                12: "dinophysis caudata",
                13: "ditylum",
                14: "eucampia",
                15: "guinardia flaccida",
                16: "helicotheca",
                17: "navicula",
                18: "pleurosigma pelagicum",
                19: "protoperidinium",
                20: "rhizosolenia",
                21: "skeletonema",
                22: "thalassionema frauenfeldii",
                23: "thalassionema nitzschioides",
            }

        # Inverted mapping: name -> id
        self.name_to_id = {v.strip().lower(): k for k, v in self.class_names.items()}

    def _parse_xml(self, xml_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
        """
        Parses a single VOC XML file and converts coordinates to normalized YOLO format.
        Returns: (width, height, [(class_id, x_center, y_center, w, h)])
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image size
        size_elem = root.find("size")
        width = int(size_elem.find("width").text) if size_elem is not None and size_elem.find("width") is not None else 0
        height = int(size_elem.find("height").text) if size_elem is not None and size_elem.find("height") is not None else 0

        # Fallback to reading image directly if size in XML is invalid
        if width <= 0 or height <= 0:
            stem = xml_path.stem
            img_file = self.images_dir / f"{stem}.jpg"
            if img_file.exists():
                with Image.open(img_file) as img:
                    width, height = img.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Could not determine dimensions for {xml_path}")

        boxes = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            cls_name = name_elem.text.strip().lower()
            if cls_name not in self.name_to_id:
                continue

            cls_id = self.name_to_id[cls_name]
            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # Coordinate clamping
            xmin = max(0.0, min(float(width), xmin))
            xmax = max(0.0, min(float(width), xmax))
            ymin = max(0.0, min(float(height), ymin))
            ymax = max(0.0, min(float(height), ymax))

            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue

            # Convert to YOLO (x_center, y_center, width, height) normalized [0, 1]
            x_center = (xmin + bw / 2.0) / width
            y_center = (ymin + bh / 2.0) / height
            norm_w = bw / width
            norm_h = bh / height

            # Clamp normalized
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            boxes.append((cls_id, x_center, y_center, norm_w, norm_h))

        return width, height, boxes

    def _read_split_ids(self, split_name: str) -> List[str]:
        """Reads image stems from split txt file if exists."""
        split_file = self.imagesets_dir / f"{split_name}.txt"
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def convert(
        self,
        mode: str = "full",
        mini_samples: int = 40,
        link_mode: str = "copy",
        seed: int = 42,
    ) -> Path:
        """
        Converts dataset to YOLO format.
        Args:
            mode: 'full' for all images, or 'mini' for a fast small sample.
            mini_samples: Total number of samples when mode == 'mini'.
            link_mode: 'copy', 'symlink', or 'hardlink'.
            seed: random seed for sampling.
        Returns:
            Path to created data.yaml
        """
        self.logger.info(f"Starting conversion | Mode: {mode} | Target: {self.output_dir}")
        random.seed(seed)

        # Output folder structure
        images_out = self.output_dir / "images"
        labels_out = self.output_dir / "labels"

        for split in ["train", "val", "test"]:
            (images_out / split).mkdir(parents=True, exist_ok=True)
            (labels_out / split).mkdir(parents=True, exist_ok=True)

        splits = {
            "train": self._read_split_ids("train"),
            "val": self._read_split_ids("val"),
            "test": self._read_split_ids("test"),
        }

        # Fallback if split files are missing
        if not splits["train"]:
            self.logger.warning("Split files not found in ImageSets/Main. Creating 70/15/15 random split.")
            all_xmls = [p.stem for p in self.annot_dir.glob("*.xml")]
            random.shuffle(all_xmls)
            n = len(all_xmls)
            splits["train"] = all_xmls[: int(n * 0.7)]
            splits["val"] = all_xmls[int(n * 0.7) : int(n * 0.85)]
            splits["test"] = all_xmls[int(n * 0.85) :]

        # Apply mini sample filter if requested
        if mode == "mini":
            train_cnt = max(1, int(mini_samples * 0.6))
            val_cnt = max(1, int(mini_samples * 0.2))
            test_cnt = max(1, int(mini_samples * 0.2))

            splits["train"] = random.sample(splits["train"], min(train_cnt, len(splits["train"])))
            splits["val"] = random.sample(splits["val"], min(val_cnt, len(splits["val"])))
            splits["test"] = random.sample(splits["test"], min(test_cnt, len(splits["test"])))
            self.logger.info(f"Mini dataset selected: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

        total_converted = 0
        total_boxes = 0

        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False

        for split, stems in splits.items():
            self.logger.info(f"Processing split '{split}': {len(stems)} items")
            iterator = tqdm(stems, desc=f"Converting {split}") if has_tqdm else stems

            for idx, stem in enumerate(iterator):
                img_path = self.images_dir / f"{stem}.jpg"
                xml_path = self.annot_dir / f"{stem}.xml"

                if not img_path.exists() or not xml_path.exists():
                    continue

                try:
                    _, _, boxes = self._parse_xml(xml_path)
                except Exception as e:
                    self.logger.warning(f"Error parsing {xml_path}: {e}")
                    continue

                # Write YOLO label txt
                lbl_out_path = labels_out / split / f"{stem}.txt"
                with open(lbl_out_path, "w", encoding="utf-8") as lf:
                    for b in boxes:
                        lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                # Copy or link image
                img_out_path = images_out / split / f"{stem}.jpg"
                if not img_out_path.exists():
                    if link_mode == "symlink":
                        try:
                            os.symlink(img_path.resolve(), img_out_path)
                        except OSError:
                            shutil.copy2(img_path, img_out_path)
                    elif link_mode == "hardlink":
                        try:
                            os.link(img_path.resolve(), img_out_path)
                        except OSError:
                            shutil.copy2(img_path, img_out_path)
                    else:
                        shutil.copy2(img_path, img_out_path)

                total_converted += 1
                total_boxes += len(boxes)

        # Generate data.yaml for YOLO
        # Use relative paths from output_dir
        data_yaml = {
            "path": str(self.output_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.class_names),
            "names": {int(k): str(v) for k, v in self.class_names.items()},
        }
        yaml_path = self.output_dir / "data.yaml"
        save_yaml(data_yaml, yaml_path)

        self.logger.info(
            f"Conversion completed successfully!\n"
            f"  Images converted: {total_converted}\n"
            f"  Bounding boxes: {total_boxes}\n"
            f"  YOLO dataset config: {yaml_path}"
        )
        return yaml_path
