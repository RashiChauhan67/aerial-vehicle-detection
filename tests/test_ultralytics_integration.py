"""Opt-in end-to-end test: trains and validates a tiny YOLO model on a synthetic dataset.

Needs Ultralytics and network access (to download ``yolov8n.pt``), so it is skipped unless
``RUN_YOLO_INTEGRATION=1`` is set.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_YOLO_INTEGRATION") != "1", reason="set RUN_YOLO_INTEGRATION=1 to run"
)


def test_train_validate_and_predict(dataset_root, tmp_path):
    from vehicle_detection.dataset import build_data_yaml
    from vehicle_detection.evaluation import validate
    from vehicle_detection.inference import predict
    from vehicle_detection.training import TrainConfig, train

    cfg = TrainConfig(
        model="yolov8n.pt",
        epochs=1,
        imgsz=64,
        batch=2,
        project=str(tmp_path / "runs"),
        name="t",
        device="cpu",
    )
    run_dir = train(dataset_root, cfg)
    weights = run_dir / "weights" / "best.pt"
    assert weights.exists()

    data_yaml = build_data_yaml(dataset_root, tmp_path / "data.yaml")
    summary = validate(weights, data_yaml, "val", imgsz=64)
    assert set(summary["per_class"]) >= {"sedan", "trailer"}

    out = predict(weights, dataset_root / "test" / "images", tmp_path / "pred")
    assert Path(out).exists()
