"""Run a trained detector on images and draw its predictions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import CLASS_NAMES
from .training import _load_yolo


def predict(
    weights: str | Path,
    source: str | Path,
    output_dir: str | Path = "runs/predict",
    conf: float = 0.25,
) -> Path:
    """Detect vehicles in ``source`` (an image or folder).

    Saves annotated images and YOLO ``.txt`` label files **with confidence scores** under
    ``output_dir/labels``. Returns the output directory.
    """
    model = _load_yolo()(str(weights))
    results = model.predict(
        source=str(source),
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(Path(output_dir).parent),
        name=Path(output_dir).name,
        exist_ok=True,
        verbose=False,
    )
    return Path(results[0].save_dir) if results else Path(output_dir)


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    """Convert a normalised YOLO box to pixel corner coordinates ``(x1, y1, x2, y2)``."""
    x1 = int(round((x_center - width / 2) * image_w))
    y1 = int(round((y_center - height / 2) * image_h))
    x2 = int(round((x_center + width / 2) * image_w))
    y2 = int(round((y_center + height / 2) * image_h))
    return x1, y1, x2, y2


def read_label_file(path: str | Path) -> list[tuple[int, float, float, float, float, float | None]]:
    """Parse a YOLO label file. Each row is ``(class, xc, yc, w, h, confidence_or_None)``."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) not in (5, 6):
            continue  # skip malformed lines
        cls, xc, yc, w, h = int(float(parts[0])), *map(float, parts[1:5])
        rows.append((cls, xc, yc, w, h, float(parts[5]) if len(parts) == 6 else None))
    return rows


def draw_detections(
    image_bgr: np.ndarray, label_file: str | Path, class_names: tuple[str, ...] = CLASS_NAMES
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the boxes from ``label_file`` drawn on it."""
    out = image_bgr.copy()
    height, width = out.shape[:2]
    for cls, xc, yc, w, h, conf in read_label_file(label_file):
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, width, height)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = class_names[cls] if conf is None else f"{class_names[cls]} {conf:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
