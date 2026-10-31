"""Print the number of labelled objects per class in each split and warn about empty classes.

Example:
    python scripts/check_dataset.py --dataset-root /path/to/aerial_images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_detection.dataset import check_split_coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    report = check_split_coverage(args.dataset_root)
    print(json.dumps(report["instances"], indent=2))
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
