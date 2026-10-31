"""Fine-tune a YOLOv8 detector on the dataset. Ultralytics is imported lazily."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import CLASS_NAMES, build_data_yaml


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "Ultralytics is required for training and inference: pip install ultralytics"
        ) from exc
    return YOLO


@dataclass
class TrainConfig:
    """Training settings. Defaults match the notebook."""

    model: str = "yolov8m.pt"
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    project: str = "runs"
    name: str = "vaid-train"
    device: str | None = None


def train(dataset_root: str | Path, cfg: TrainConfig | None = None) -> Path:
    """Train and return the run directory (it holds ``weights/best.pt`` and ``results.csv``)."""
    cfg = cfg or TrainConfig()
    data_yaml = build_data_yaml(
        dataset_root, Path(cfg.project) / cfg.name / "data.yaml", CLASS_NAMES
    )
    model = _load_yolo()(cfg.model)
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        project=cfg.project,
        name=cfg.name,
        device=cfg.device,
    )
    return Path(results.save_dir)
