#!/usr/bin/env python3
"""Copy the public SEEK training data into LLaMA-Factory and register it."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict


DATASETS = {
    "musique_decomp": Path("musique/train.json"),
    "twowiki_decomp": Path("twowiki/train.json"),
    "popqa_decomp": Path("popqa/train.json"),
}


def _validate_training_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing training file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Training file must contain a non-empty JSON list: {path}")


def prepare_data(llamafactory_dir: Path, dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    """Copy the three training files and update LLaMA-Factory dataset_info.json."""
    llamafactory_dir = llamafactory_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    dataset_info_path = llamafactory_dir / "data" / "dataset_info.json"
    source_root = dataset_dir / "data"
    target_root = llamafactory_dir / "data" / "seek_decomposer_12k"

    if not dataset_info_path.is_file():
        raise FileNotFoundError(
            f"LLaMA-Factory dataset registry was not found: {dataset_info_path}"
        )

    registrations: Dict[str, Dict[str, str]] = {}
    for dataset_name, relative_path in DATASETS.items():
        source_path = source_root / relative_path
        _validate_training_file(source_path)

        target_path = target_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        registrations[dataset_name] = {
            "file_name": str(Path("seek_decomposer_12k") / relative_path)
        }

    with dataset_info_path.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)
    if not isinstance(dataset_info, dict):
        raise ValueError(f"Expected a JSON object in {dataset_info_path}")

    dataset_info.update(registrations)
    temporary_path = dataset_info_path.with_name("dataset_info.json.seek-tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(dataset_info_path)

    return registrations


def validate_config(config_path: Path) -> None:
    """Parse the YAML with the pinned LLaMA-Factory argument definitions."""
    from omegaconf import OmegaConf
    from llamafactory.hparams.parser import _parse_train_args

    config = OmegaConf.to_container(OmegaConf.load(config_path.resolve()))
    model_args, data_args, training_args, finetuning_args, _ = _parse_train_args(config)

    expected_datasets = list(DATASETS)
    if data_args.dataset != expected_datasets:
        raise ValueError(
            f"Unexpected dataset list: {data_args.dataset}; expected {expected_datasets}"
        )
    if finetuning_args.stage != "sft" or finetuning_args.finetuning_type != "lora":
        raise ValueError("The decomposer configuration must use LoRA SFT.")

    print("Validated LLaMA-Factory configuration:")
    print(f"  model: {model_args.model_name_or_path}")
    print(f"  datasets: {','.join(data_args.dataset)}")
    print(f"  output: {training_args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the SEEK 12K dataset for a LLaMA-Factory checkout."
    )
    parser.add_argument(
        "--llamafactory-dir",
        type=Path,
        required=True,
        help="Path to the pinned LLaMA-Factory checkout.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path produced by `hf download Chien1011/SEEK-decomposer-12k`.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional training YAML to validate with LLaMA-Factory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registrations = prepare_data(args.llamafactory_dir, args.dataset_dir)
    print("Prepared LLaMA-Factory datasets:")
    for dataset_name, metadata in registrations.items():
        print(f"  {dataset_name}: {metadata['file_name']}")
    if args.config is not None:
        validate_config(args.config)


if __name__ == "__main__":
    main()
