from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..utils.helpers import get_device, load_yaml, setup_logger


class YOLO26Trainer:
    """Encapsulates training workflows for YOLO26 detection models."""

    def __init__(
        self,
        model_name_or_path: str = "yolo26n.pt",
        config_path: Optional[Union[str, Path]] = None,
        logger=None,
    ):
        self.logger = logger or setup_logger("YOLO26Trainer")
        self.model_name_or_path = str(model_name_or_path)
        self.config_path = Path(config_path) if config_path else None
        self.config = load_yaml(self.config_path) if self.config_path else {}

    def train(self, **overrides: Any) -> Any:
        """
        Executes model training with merged configuration and overrides.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is not installed. Run 'pip install ultralytics' first."
            )

        # Merge config file options with runtime keyword arguments
        train_args = {**self.config, **overrides}

        # Resolve model name
        model_name = train_args.pop("model", self.model_name_or_path)
        self.logger.info(f"Initializing YOLO model: {model_name}")
        model = YOLO(model_name)

        # Resolve dataset yaml path
        data_yaml = train_args.get("data")
        if data_yaml:
            resolved_data = Path(data_yaml).resolve()
            if not resolved_data.is_file():
                self.logger.warning(f"Data YAML {resolved_data} not found on disk, passing as-is.")
            else:
                train_args["data"] = str(resolved_data).replace("\\", "/")

        # Resolve device
        preferred_dev = train_args.get("device")
        train_args["device"] = get_device(preferred_dev)

        self.logger.info(f"Training parameters: epochs={train_args.get('epochs', 100)}, batch={train_args.get('batch', 16)}, device={train_args['device']}")

        # Start training
        results = model.train(**train_args)

        best_weights = getattr(model.trainer, "best", None)
        self.logger.info(f"Training completed successfully! Best weights: {best_weights}")
        return results
