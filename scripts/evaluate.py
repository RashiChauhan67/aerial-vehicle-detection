"""Validate a trained detector on the validation or test split and print the metrics.

Example:
    python scripts/evaluate.py --weights runs/vaid-train/weights/best.pt \\
        --dataset-root /path/to/aerial_images --split val
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from vehicle_detection.dataset import build_data_yaml, check_split_coverage
from vehicle_detection.evaluation import validate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    for warning in check_split_coverage(args.dataset_root)["warnings"]:
        print(f"WARNING: {warning}")
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = build_data_yaml(args.dataset_root, Path(tmp) / "data.yaml")
        print(json.dumps(validate(args.weights, data_yaml, args.split), indent=2))


if __name__ == "__main__":
    main()
