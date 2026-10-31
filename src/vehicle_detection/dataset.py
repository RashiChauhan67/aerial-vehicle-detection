"""Dataset configuration and sanity checks for a YOLO-format aerial vehicle dataset.

Expected layout (the layout of the Kaggle VAID copy used in the notebook)::

    <root>/
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    └── test/images,  test/labels

Label files are in YOLO format: ``<class> <x_center> <y_center> <width> <height>`` (normalised).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CLASS_NAMES: tuple[str, ...] = (
    "sedan",
    "minibus",
    "truck",
    "pickup",
    "bus",
    "cement truck",
    "trailer",
)
SPLITS: dict[str, str] = {"train": "train", "val": "valid", "test": "test"}


def build_data_yaml(
    dataset_root: str | Path,
    output_path: str | Path,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> Path:
    """Write the ``data.yaml`` Ultralytics needs, with absolute image paths.

    Raises:
        FileNotFoundError: if a split's ``images`` folder does not exist.
    """
    root = Path(dataset_root).resolve()
    config: dict = {}
    for key, folder in SPLITS.items():
        images = root / folder / "images"
        if not images.is_dir():
            raise FileNotFoundError(f"Missing image folder for the {key!r} split: {images}")
        config[key] = str(images)
    config["nc"] = len(class_names)
    config["names"] = list(class_names)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return output_path


def count_instances(labels_dir: str | Path, num_classes: int = len(CLASS_NAMES)) -> list[int]:
    """Count labelled objects per class in a folder of YOLO ``.txt`` files."""
    counts = [0] * num_classes
    for path in sorted(Path(labels_dir).glob("*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            if not 0 <= class_id < num_classes:
                raise ValueError(f"Class id {class_id} out of range in {path}")
            counts[class_id] += 1
    return counts


def check_split_coverage(
    dataset_root: str | Path, class_names: tuple[str, ...] = CLASS_NAMES
) -> dict:
    """Instances per class in every split, plus warnings for classes missing from a split.

    A class with no instances in the validation or test split cannot be evaluated, and the
    detector's reported per-class metrics for it are meaningless.
    """
    root = Path(dataset_root)
    per_split: dict[str, dict[str, int]] = {}
    for key, folder in SPLITS.items():
        labels = root / folder / "labels"
        counts = count_instances(labels, len(class_names)) if labels.is_dir() else None
        per_split[key] = dict(zip(class_names, counts, strict=True)) if counts else {}

    warnings = [
        f"class {name!r} has no instances in the {key} split"
        for key in ("val", "test")
        for name, n in per_split[key].items()
        if n == 0
    ]
    for key in ("val", "test"):
        if not per_split[key]:
            warnings.append(f"no label files found for the {key} split")
    return {"instances": per_split, "warnings": warnings}
