import numpy as np

from vehicle_detection.evaluation import summarize_metrics
from vehicle_detection.inference import draw_detections, read_label_file, yolo_to_xyxy


def test_yolo_to_xyxy():
    assert yolo_to_xyxy(0.5, 0.5, 0.5, 0.25, image_w=200, image_h=100) == (50, 38, 150, 62)


def test_read_label_file_handles_confidence_and_skips_bad_lines(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.2 0.2 0.91\n1 0.1 0.2 0.3 0.4\nbad line\n")
    rows = read_label_file(path)
    assert rows == [(0, 0.5, 0.5, 0.2, 0.2, 0.91), (1, 0.1, 0.2, 0.3, 0.4, None)]


def test_draw_detections_changes_pixels_and_keeps_the_input(tmp_path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    label = tmp_path / "a.txt"
    label.write_text("0 0.5 0.5 0.4 0.4 0.9\n")

    drawn = draw_detections(image, label)

    assert drawn.shape == image.shape and drawn.sum() > 0
    assert image.sum() == 0  # the input image is untouched


class _FakeBox:
    """Stands in for ultralytics' Metrics.box with two of the seven classes present."""

    ap_class_index = np.array([0, 4])
    mp, mr, map50, map = 0.9, 0.95, 0.99, 0.8

    def class_result(self, i):
        return [(0.8, 0.9, 0.97, 0.7), (1.0, 1.0, 0.995, 0.99)][i]


def test_summarize_metrics_marks_absent_classes():
    summary = summarize_metrics(_FakeBox())

    assert summary["map50_95"] == 0.8
    assert summary["per_class"]["sedan"] == {
        "present": True,
        "precision": 0.8,
        "recall": 0.9,
        "map50": 0.97,
        "map50_95": 0.7,
    }
    assert summary["per_class"]["bus"]["map50_95"] == 0.99
    assert summary["per_class"]["trailer"] == {"present": False}
