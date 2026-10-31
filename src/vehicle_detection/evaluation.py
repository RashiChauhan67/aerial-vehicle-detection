"""Validate a trained detector and summarise the metrics."""

from __future__ import annotations

from pathlib import Path

from .dataset import CLASS_NAMES
from .training import _load_yolo


def summarize_metrics(box, class_names: tuple[str, ...] = CLASS_NAMES) -> dict:
    """Turn an Ultralytics ``Metrics.box`` object into plain Python numbers.

    ``box.ap_class_index`` lists only the classes that actually occur in the evaluated split;
    classes absent from the split are reported with ``"present": False`` and no metrics.
    """
    present = {int(i) for i in box.ap_class_index}
    per_class = {}
    for pos, class_id in enumerate(sorted(present)):
        precision, recall, ap50, ap = box.class_result(pos)
        per_class[class_names[class_id]] = {
            "present": True,
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(ap50),
            "map50_95": float(ap),
        }
    for name in class_names:
        per_class.setdefault(name, {"present": False})
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
    }


def validate(
    weights: str | Path, data_yaml: str | Path, split: str = "val", imgsz: int = 640
) -> dict:
    """Run Ultralytics validation on ``split`` (``val`` or ``test``) and summarise the result."""
    model = _load_yolo()(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, verbose=False)
    return summarize_metrics(metrics.box)
