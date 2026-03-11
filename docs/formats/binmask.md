# Binary Mask Format

The **Binary Mask** (BinMask) format stores segmentation labels as binary PNG images where each pixel value encodes a class ID.

---

## Directory Structure

```
binmask_dset/
├── images/
│   ├── img_01.png
│   ├── img_02.png
│   └── ...
├── labels/
│   ├── img_01.png
│   ├── img_02.png
│   └── ...
└── classes.txt
```

### Components

| Component | Description |
|-----------|-------------|
| `images/` | Labelled images in `jpg` or `png` format. |
| `labels/` | One binary `png` mask per image. Each pixel's intensity value corresponds to a class ID. |
| `classes.txt` | Text file listing class names, one per line. The line index corresponds to the class ID used in the masks. |

---

## Mask Encoding

In the label images:

- A pixel value of `0` represents **background**.
- A pixel value of `255` represents **foreground** (for single-class binary masks).
- For multi-class masks, each pixel value corresponds to a class index from `classes.txt`.

---

## Supported Tasks

| Task | Supported |
|------|:---------:|
| Segmentation | ✅ |
| Object Detection | ✅ (via contour extraction) |

---

## Available Conversions

| Direction | Function |
|-----------|----------|
| BinMask → YOLO | `binmask_to_yolo()` |
<!-- | YOLO → BinMask | `yolo_to_binmask()` | -->
| BinMask → Label Studio | `binmask_to_ls()` |

See the [Conversions Usage Guide](../conversions/usage.md) for CLI and API examples.
