import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_llamafactory_data.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_llamafactory_data", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareLlamaFactoryDataTest(unittest.TestCase):
    def _make_layout(self, root: Path) -> tuple[Path, Path]:
        llamafactory_dir = root / "LLaMA-Factory"
        dataset_dir = root / "SEEK-decomposer-12k"
        (llamafactory_dir / "data").mkdir(parents=True)
        (llamafactory_dir / "data" / "dataset_info.json").write_text(
            json.dumps({"existing": {"file_name": "existing.json"}}),
            encoding="utf-8",
        )

        for relative_path in MODULE.DATASETS.values():
            training_path = dataset_dir / "data" / relative_path
            training_path.parent.mkdir(parents=True, exist_ok=True)
            training_path.write_text(
                json.dumps([{"instruction": "q", "output": "a"}]),
                encoding="utf-8",
            )
        return llamafactory_dir, dataset_dir

    def test_prepare_data_copies_and_registers_all_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            llamafactory_dir, dataset_dir = self._make_layout(root)

            registrations = MODULE.prepare_data(llamafactory_dir, dataset_dir)

            self.assertEqual(set(registrations), set(MODULE.DATASETS))
            registry = json.loads(
                (llamafactory_dir / "data" / "dataset_info.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("existing", registry)
            for dataset_name, relative_path in MODULE.DATASETS.items():
                copied_path = (
                    llamafactory_dir / "data" / "seek_decomposer_12k" / relative_path
                )
                self.assertTrue(copied_path.is_file())
                self.assertEqual(
                    registry[dataset_name]["file_name"],
                    str(Path("seek_decomposer_12k") / relative_path),
                )

    def test_prepare_data_rejects_missing_training_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            llamafactory_dir, dataset_dir = self._make_layout(root)
            (dataset_dir / "data" / "popqa" / "train.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "popqa/train.json"):
                MODULE.prepare_data(llamafactory_dir, dataset_dir)


if __name__ == "__main__":
    unittest.main()
