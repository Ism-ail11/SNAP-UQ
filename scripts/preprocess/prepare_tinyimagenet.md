# TinyImageNet prep

1) Download Tiny ImageNet (200 classes) and extract into:
   `data/tinyimagenet/`

Expected files:
- `data/tinyimagenet/wnids.txt`
- `data/tinyimagenet/train/<wnid>/images/*.JPEG`
- `data/tinyimagenet/val/images/*.JPEG`
- `data/tinyimagenet/val/val_annotations.txt`

2) (Optional) Put TinyImageNet-C in:
   `data/tinyimagenet_c/` as `corruption/severity/*.npy` or your own format.
   You can also use on-the-fly corruptions via `snapuq/data/corruptions/vision_corruptions.py`.
