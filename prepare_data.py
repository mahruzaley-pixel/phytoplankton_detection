import argparse
from pathlib import Path
import sys

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).parent))

from src.data.voc_to_yolo import VOCtoYOLOConverter
from src.data.validator import DatasetValidator
from src.utils.helpers import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Pascal VOC Phytoplankton dataset to YOLO format.")
    parser.add_argument("--raw_dir", type=str, default="dataset", help="Path to raw VOC dataset directory.")
    parser.add_argument("--output_dir", type=str, default="data/yolo", help="Destination directory for YOLO dataset.")
    parser.add_argument("--mode", type=str, choices=["full", "mini"], default="full", help="'full' or 'mini' (small sample for local testing).")
    parser.add_argument("--mini_samples", type=int, default=40, help="Total sample count when mode is 'mini'.")
    parser.add_argument("--link_mode", type=str, choices=["copy", "symlink", "hardlink"], default="copy", help="File linking mode.")
    parser.add_argument("--classes_config", type=str, default="configs/classes.yaml", help="Path to classes.yaml.")
    parser.add_argument("--no_validate", action="store_true", help="Skip dataset validation after conversion.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("prepare_data")

    logger.info("=" * 60)
    logger.info("Phytoplankton Dataset Converter (VOC -> YOLO)")
    logger.info(f"Raw Directory   : {args.raw_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Mode            : {args.mode}")
    logger.info("=" * 60)

    converter = VOCtoYOLOConverter(
        raw_dataset_dir=args.raw_dir,
        output_dir=args.output_dir,
        classes_config_path=args.classes_config,
        logger=logger,
    )

    yaml_path = converter.convert(
        mode=args.mode,
        mini_samples=args.mini_samples,
        link_mode=args.link_mode,
    )

    if not args.no_validate:
        logger.info("\nRunning Dataset Validation...")
        validator = DatasetValidator(yaml_path, logger=logger)
        validator.validate()

    logger.info(f"\n[DONE] Dataset is ready at: {args.output_dir}")
    logger.info(f"Config path for YOLO training: {yaml_path}")


if __name__ == "__main__":
    main()
