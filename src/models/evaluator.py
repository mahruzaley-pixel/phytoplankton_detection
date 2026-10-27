from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..utils.helpers import get_device, setup_logger


class YOLO26Evaluator:
    """Evaluates trained YOLO26 model checkpoints on validation or test sets."""

    def __init__(self, weights_path: Union[str, Path], logger=None):
        self.weights_path = Path(weights_path)
        self.logger = logger or setup_logger("YOLO26Evaluator")

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

    def evaluate(
        self,
        data_yaml: Union[str, Path],
        split: str = "val",
        batch: int = 16,
        imgsz: int = 800,
        device: Optional[Union[str, int]] = None,
        save_json: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs validation or testing on the given dataset split.
        split: 'val' or 'test'
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics is not installed. Run 'pip install ultralytics'.")

        data_yaml_path = Path(data_yaml).resolve()
        if not data_yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        chosen_device = get_device(device)
        self.logger.info(f"Loading checkpoint: {self.weights_path}")
        model = YOLO(str(self.weights_path))

        self.logger.info(f"Running evaluation on '{split}' split (imgsz={imgsz}, batch={batch}, device={chosen_device})...")
        metrics = model.val(
            data=str(data_yaml_path).replace("\\", "/"),
            split=split,
            batch=batch,
            imgsz=imgsz,
            device=chosen_device,
            save_json=save_json,
        )

        map50 = getattr(metrics.box, "map50", 0.0)
        map50_95 = getattr(metrics.box, "map", 0.0)
        mp = getattr(metrics.box, "mp", 0.0)
        mr = getattr(metrics.box, "mr", 0.0)

        self.logger.info(
            f"Evaluation Results on [{split}]:\n"
            f"  mAP@50     : {map50:.4f}\n"
            f"  mAP@50-95  : {map50_95:.4f}\n"
            f"  Precision  : {mp:.4f}\n"
            f"  Recall     : {mr:.4f}"
        )

        return {
            "mAP50": float(map50),
            "mAP50_95": float(map50_95),
            "precision": float(mp),
            "recall": float(mr),
            "raw_metrics": metrics,
        }
