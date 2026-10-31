import cv2
import numpy as np
import pytest

from vehicle_detection.dataset import CLASS_NAMES


def _write_split(root, folder, rows_per_image, size=(64, 96), n_images=None):
    images = root / folder / "images"
    labels = root / folder / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i, rows in enumerate(rows_per_image):
        cv2.imwrite(str(images / f"{i}.jpg"), rng.integers(0, 255, (*size, 3), dtype=np.uint8))
        (labels / f"{i}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))


@pytest.fixture
def dataset_root(tmp_path):
    """A tiny YOLO dataset where 'trailer' (class 6) never appears in valid/test."""
    box = "{} 0.5 0.5 0.2 0.2"
    _write_split(tmp_path, "train", [[box.format(c)] for c in range(len(CLASS_NAMES))] * 2)
    _write_split(tmp_path, "valid", [[box.format(0), box.format(1)], [box.format(0)]])
    _write_split(tmp_path, "test", [[box.format(c)] for c in range(6)])
    return tmp_path
