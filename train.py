import argparse
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).parent))

from src.models.trainer import YOLO26Trainer
from src.utils.helpers import setup_logger, load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO26 on Phytoplankton dataset.")
    parser.add_argument("--config", type=str, default="configs/train_kaggle.yaml", help="Path to training config YAML.")
    parser.add_argument("--model", type=str, default=None, help="YOLO model checkpoint/name (e.g. yolo26n.pt, yolo26s.pt).")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. '0', 'cpu', '0,1').")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (subfolder inside runs/train).")
    parser.add_argument("--project", type=str, default=None, help="Project directory (default: runs/train).")
    parser.add_argument("--local_test", action="store_true", help="Shortcut for rapid local test using configs/train_local.yaml.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("train")

    config_path = "configs/train_local.yaml" if args.local_test else args.config

    logger.info("=" * 60)
    logger.info("YOLO26 Phytoplankton Detection Training Pipeline")
    logger.info(f"Base Config: {config_path}")
    if args.local_test:
        logger.info("Mode       : LOCAL TEST (Fast execution)")
    logger.info("=" * 60)

    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.data:
        overrides["data"] = args.data
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device
    if args.name is not None:
        overrides["name"] = args.name
    if args.project is not None:
        overrides["project"] = args.project

    trainer = YOLO26Trainer(
        model_name_or_path=overrides.get("model", "yolo26n.pt" if args.local_test else "yolo26s.pt"),
        config_path=config_path,
        logger=logger,
    )

    results = trainer.train(**overrides)
    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
