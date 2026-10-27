import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.evaluator import YOLO26Evaluator
from src.utils.helpers import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 model on validation or test set.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt).")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution.")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu').")
    parser.add_argument("--save_json", action="store_true", help="Save predictions to JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("eval")

    logger.info("=" * 60)
    logger.info("YOLO26 Phytoplankton Evaluation")
    logger.info(f"Weights : {args.weights}")
    logger.info(f"Dataset : {args.data}")
    logger.info(f"Split   : {args.split}")
    logger.info("=" * 60)

    evaluator = YOLO26Evaluator(weights_path=args.weights, logger=logger)
    metrics = evaluator.evaluate(
        data_yaml=args.data,
        split=args.split,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
    )

    logger.info(f"Finished evaluation on {args.split} set.")


if __name__ == "__main__":
    main()
