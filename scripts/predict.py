"""Detect vehicles in an image or folder of images.

Example:
    python scripts/predict.py --weights runs/vaid-train/weights/best.pt --source images/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_detection.inference import predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/predict"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    print(f"Saved predictions to {predict(args.weights, args.source, args.output_dir, args.conf)}")


if __name__ == "__main__":
    main()
