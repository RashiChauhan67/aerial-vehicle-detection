import pytest
import yaml

from vehicle_detection.dataset import (
    CLASS_NAMES,
    build_data_yaml,
    check_split_coverage,
    count_instances,
)


def test_build_data_yaml_has_absolute_paths_and_classes(dataset_root, tmp_path):
    out = build_data_yaml(dataset_root, tmp_path / "out" / "data.yaml")
    config = yaml.safe_load(out.read_text())

    assert config["nc"] == 7 and config["names"] == list(CLASS_NAMES)
    assert config["train"].endswith("train/images") and config["val"].endswith("valid/images")
    assert all(config[k].startswith("/") for k in ("train", "val", "test"))


def test_build_data_yaml_missing_split_raises(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_data_yaml(tmp_path, tmp_path / "data.yaml")


def test_count_instances(dataset_root):
    assert count_instances(dataset_root / "valid" / "labels") == [2, 1, 0, 0, 0, 0, 0]


def test_count_instances_rejects_unknown_class(tmp_path):
    (tmp_path / "a.txt").write_text("9 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError):
        count_instances(tmp_path)


def test_check_split_coverage_warns_about_classes_missing_from_val_and_test(dataset_root):
    report = check_split_coverage(dataset_root)

    assert report["instances"]["train"]["trailer"] == 2
    assert "class 'trailer' has no instances in the test split" in report["warnings"]
    assert "class 'truck' has no instances in the val split" in report["warnings"]
    assert not any("train" in w for w in report["warnings"])


def test_check_split_coverage_flags_missing_labels(tmp_path):
    report = check_split_coverage(tmp_path)
    assert "no label files found for the val split" in report["warnings"]
