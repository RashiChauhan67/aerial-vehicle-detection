# Data

No data is stored in this repository. Download the dataset yourself and check its license and terms before use.

## Dataset

**VAID** (Vehicle Aerial Imaging from Drone): top-down drone images of roads with vehicle boxes. The notebook read a Kaggle copy (slug `vaid-dataset`) that is already in YOLO format.

Expected layout:

```
aerial_images/
├── train/  images/  labels/
├── valid/  images/  labels/
└── test/   images/  labels/
```

Labels are YOLO text files, one per image, one row per vehicle: `<class> <x_center> <y_center> <width> <height>`, all normalised to [0, 1].

Classes, in label order: `0 sedan`, `1 minibus`, `2 truck`, `3 pickup`, `4 bus`, `5 cement truck`, `6 trailer`.

## Sizes seen in the notebook

- Validation split: 49 images, 152 labelled vehicles (124 sedans, 15 minibuses, 7 pickups, 5 trucks, 1 bus; **no cement trucks or trailers**).
- Test split: 50 images.

Run `python scripts/check_dataset.py --dataset-root aerial_images` to count instances per class in each split of your copy.

## Source and citation

VAID is described in: H.-Y. Lin, K.-C. Tu, C.-Y. Li. *VAID: An Aerial Image Dataset for Vehicle Detection and Classification.* IEEE Access, 2020. Please verify the citation and the license shown on the dataset page for your copy before reuse, and cite it when you publish results.
