import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.predictor import YOLO26Predictor
from src.utils.helpers import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO26 inference on phytoplankton images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt).")
    parser.add_argument("--source", type=str, required=True, help="Path to input image, folder of images, or glob.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=800, help="Image resolution.")
    parser.add_argument("--save_dir", type=str, default="runs/predict", help="Directory to save visualization results.")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cpu').")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("predict")

    logger.info("=" * 60)
    logger.info("YOLO26 Phytoplankton Inference")
    logger.info(f"Weights   : {args.weights}")
    logger.info(f"Source    : {args.source}")
    logger.info(f"Output Dir: {args.save_dir}")
    logger.info("=" * 60)

    predictor = YOLO26Predictor(weights_path=args.weights, device=args.device, logger=logger)
    results = predictor.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=True,
        save_dir=args.save_dir,
    )

    logger.info(f"Predictions saved to {args.save_dir}")


if __name__ == "__main__":
    main()
