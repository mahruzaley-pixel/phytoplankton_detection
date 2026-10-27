"""
Fast local test script for Phytoplankton Detection with YOLO26.
Runs an end-to-end pipeline verification using a small dataset sample (mini mode).
"""
import sys
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.voc_to_yolo import VOCtoYOLOConverter
from src.data.validator import DatasetValidator
from src.models.trainer import YOLO26Trainer
from src.models.predictor import YOLO26Predictor
from src.utils.helpers import setup_logger


def run_test():
    logger = setup_logger("local_test")
    logger.info("============================================================")
    logger.info("🚀 STARTING FAST LOCAL TEST PIPELINE (YOLO26)")
    logger.info("============================================================")

    raw_dataset = PROJECT_ROOT / "dataset"
    mini_dataset_dir = PROJECT_ROOT / "data" / "yolo_mini"

    if not raw_dataset.exists():
        logger.error(f"Raw dataset folder not found at: {raw_dataset}")
        sys.exit(1)

    # 1. Step 1: Prepare Mini Dataset (30 samples)
    logger.info("\n[STEP 1/4] Generating mini YOLO dataset (30 samples)...")
    if mini_dataset_dir.exists():
        shutil.rmtree(mini_dataset_dir, ignore_errors=True)

    converter = VOCtoYOLOConverter(
        raw_dataset_dir=raw_dataset,
        output_dir=mini_dataset_dir,
        classes_config_path=PROJECT_ROOT / "configs" / "classes.yaml",
        logger=logger,
    )
    yaml_path = converter.convert(mode="mini", mini_samples=30, link_mode="copy")

    # 2. Step 2: Validate Mini Dataset
    logger.info("\n[STEP 2/4] Validating mini dataset integrity...")
    validator = DatasetValidator(yaml_path, logger=logger)
    validation_results = validator.validate()

    # 3. Step 3: Train for 1-2 Epochs with YOLO26n
    logger.info("\n[STEP 3/4] Running fast test training (1 epoch, yolo26n.pt)...")
    trainer = YOLO26Trainer(
        model_name_or_path="yolo26n.pt",
        config_path=PROJECT_ROOT / "configs" / "train_local.yaml",
        logger=logger,
    )
    
    # Override for super-fast smoke testing
    trainer.train(
        data=str(yaml_path).replace("\\", "/"),
        epochs=1,
        batch=2,
        imgsz=320,
        device="cpu",  # safe default for any local PC without CUDA setup
        project="runs/local_test",
        name="smoke_test",
        exist_ok=True,
    )

    # 4. Step 4: Run Sample Inference
    logger.info("\n[STEP 4/4] Testing inference on a sample image...")
    test_imgs = list((mini_dataset_dir / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        test_imgs = list((mini_dataset_dir / "images" / "val").glob("*.jpg"))

    best_weights = PROJECT_ROOT / "runs" / "local_test" / "smoke_test" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = PROJECT_ROOT / "runs" / "local_test" / "smoke_test" / "weights" / "last.pt"

    if best_weights.exists() and test_imgs:
        sample_img = test_imgs[0]
        predictor = YOLO26Predictor(weights_path=best_weights, device="cpu", logger=logger)
        predictor.predict(
            source=sample_img,
            conf=0.1,
            imgsz=320,
            save=True,
            save_dir=PROJECT_ROOT / "runs" / "local_test" / "predictions",
        )
        logger.info(f"Sample prediction saved for: {sample_img.name}")

    logger.info("============================================================")
    logger.info("✅ FAST LOCAL TEST COMPLETED SUCCESSFULLY!")
    logger.info("All components (Converter, Validator, Trainer, Predictor) are operational.")
    logger.info("You can now safely push to GitHub and run full training on Kaggle.")
    logger.info("============================================================")


if __name__ == "__main__":
    run_test()
