from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.helpers import get_device, setup_logger


class YOLO26Predictor:
    """Performs inference and detection visualizations using trained YOLO26 models."""

    def __init__(
        self,
        weights_path: Union[str, Path],
        device: Optional[Union[str, int]] = None,
        logger=None,
    ):
        self.weights_path = Path(weights_path)
        self.logger = logger or setup_logger("YOLO26Predictor")

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics is not installed. Run 'pip install ultralytics'.")

        self.device = get_device(device)
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names

    def predict(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 800,
        save: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Any]:
        """
        Runs object detection on the given source image or directory.
        """
        self.logger.info(f"Running inference on {source} with conf={conf}, imgsz={imgsz}...")

        predict_kwargs = {
            "source": str(source),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "device": self.device,
            "save": save,
        }

        if save_dir:
            predict_kwargs["project"] = str(Path(save_dir).parent)
            predict_kwargs["name"] = str(Path(save_dir).name)
            predict_kwargs["exist_ok"] = True

        results = self.model.predict(**predict_kwargs)
        self.logger.info(f"Inference finished for {len(results)} images.")
        return results
