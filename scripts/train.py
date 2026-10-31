"""Fine-tune YOLOv8 on a YOLO-format aerial vehicle dataset.

Example:
    python scripts/train.py --dataset-root /path/to/aerial_images --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_detection.dataset import check_split_coverage
from vehicle_detection.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--model", default=TrainConfig.model, help="Ultralytics weights to start from."
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--imgsz", type=int, default=TrainConfig.imgsz)
    parser.add_argument("--batch", type=int, default=TrainConfig.batch)
    parser.add_argument("--project", default=TrainConfig.project)
    parser.add_argument("--name", default=TrainConfig.name)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu.")
    args = parser.parse_args()

    for warning in check_split_coverage(args.dataset_root)["warnings"]:
        print(f"WARNING: {warning}")

    cfg = TrainConfig(
        args.model, args.epochs, args.imgsz, args.batch, args.project, args.name, args.device
    )
    print(f"Run directory: {train(args.dataset_root, cfg)}")


if __name__ == "__main__":
    main()
