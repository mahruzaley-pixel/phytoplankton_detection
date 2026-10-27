"""Unit and Integration tests for phytoplankton detection pipeline."""
import unittest
import tempfile
import shutil
from pathlib import Path

from src.utils.helpers import load_yaml, save_yaml
from src.data.voc_to_yolo import VOCtoYOLOConverter
from src.data.validator import DatasetValidator


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_yaml_helpers(self):
        test_data = {"nc": 24, "names": ["a", "b"]}
        yaml_file = self.temp_path / "test.yaml"
        save_yaml(test_data, yaml_file)
        self.assertTrue(yaml_file.exists())
        loaded = load_yaml(yaml_file)
        self.assertEqual(loaded["nc"], 24)

    def test_voc_to_yolo_mini(self):
        # Test converting raw dataset in mini mode
        raw_dataset = Path("dataset")
        if not raw_dataset.exists():
            self.skipTest("dataset folder not present in working directory")

        out_dir = self.temp_path / "yolo_test"
        converter = VOCtoYOLOConverter(
            raw_dataset_dir=raw_dataset,
            output_dir=out_dir,
            classes_config_path=Path("configs/classes.yaml"),
        )
        yaml_path = converter.convert(mode="mini", mini_samples=10, link_mode="copy")
        self.assertTrue(yaml_path.exists())

        # Validate generated dataset
        validator = DatasetValidator(yaml_path)
        stats = validator.validate()
        self.assertIn("train", stats)
        self.assertGreater(stats["train"]["total_images"], 0)


if __name__ == "__main__":
    unittest.main()
