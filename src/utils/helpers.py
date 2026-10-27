import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def setup_logger(name: str = "phytoplankton", level: int = logging.INFO) -> logging.Logger:
    """Configures a standard logger with clean formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely loads a YAML configuration file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Saves a dictionary to a YAML file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def get_device(preferred_device: Optional[Union[str, int]] = None) -> Union[str, int]:
    """
    Determines the best available device (CUDA GPU or CPU),
    honoring preferred_device if available.
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if preferred_device is not None:
        if str(preferred_device).lower() == "cpu":
            return "cpu"
        if cuda_available:
            return preferred_device
        return "cpu"

    return 0 if cuda_available else "cpu"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
