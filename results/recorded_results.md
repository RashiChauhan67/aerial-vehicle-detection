# Recorded results

Everything on this page is copied from the saved outputs of the original notebook, [`../notebooks/vaid-aerial-traffic.ipynb`](../notebooks/vaid-aerial-traffic.ipynb). **None of it was re-run** for this repository.

## Setup

| Item | Value |
|---|---|
| Model | YOLOv8m, fine-tuned from `yolov8m.pt` |
| Library | Ultralytics 8.3.133, PyTorch 2.6, Python 3.11 |
| Hardware | one Tesla T4 (15 GB) |
| Training | 20 epochs, image size 640, batch size 16 |
| Fused model | 92 layers, 25,843,813 parameters, 78.7 GFLOPs |
| Speed (per image) | 0.1 ms preprocess, 7.4 ms inference, 1.1 ms postprocess |

## Validation (printed after training, best weights)

49 images, 152 labelled vehicles. Overall: precision 0.9418, recall 0.9797, mAP50 0.9948, mAP50-95 0.8510.

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 49 | 152 | 0.942 | 0.980 | 0.995 | 0.851 |
| sedan | 48 | 124 | 0.976 | 0.984 | 0.994 | 0.802 |
| minibus | 11 | 15 | 0.981 | 1.000 | 0.995 | 0.894 |
| truck | 5 | 5 | 0.950 | 1.000 | 0.995 | 0.808 |
| pickup | 5 | 7 | 1.000 | 0.915 | 0.995 | 0.757 |
| bus | 1 | 1 | 0.803 | 1.000 | 0.995 | 0.995 |

Cement truck and trailer have no instances in the validation split and are not listed by Ultralytics. The notebook's raw per-class mAP50-95 array repeats the overall value (0.85097) for those two classes, which is a placeholder, not a measurement.

## Test-image predictions

The notebook ran the trained model on the 50 test images and drew the detections for five of them (see the README). It did not compute metrics on the test split.

## Not included

The notebook's later ByteTrack "trajectory" and "count-map" steps are not part of this repository: they treated unrelated still images as consecutive video frames and fixed every confidence to 1.0.
