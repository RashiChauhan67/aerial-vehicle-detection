"""YOLOv8 vehicle detection and classification in aerial (drone) imagery."""

from .dataset import CLASS_NAMES, build_data_yaml, check_split_coverage, count_instances

__all__ = ["CLASS_NAMES", "build_data_yaml", "check_split_coverage", "count_instances"]
