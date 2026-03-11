# YOLO Format

The **YOLO** format is the standard annotation format used by YOLO models (v1–v3 and compatible). It serves as the **central hub** in AgribotTools — most conversions pass through YOLO as an intermediate format.

---

## Directory Structure

```
yolo_dset/
├── images/
│   ├── img_01.png
│   ├── img_02.png
│   └── ...
├── labels/
│   ├── img_01.txt
│   ├── img_02.txt
│   └── ...
└── classes.txt
```

### Components

| Component | Description |
|-----------|-------------|
| `images/` | Labelled images in `jpg` or `png` format. |
| `labels/` | One `.txt` file per image (same filename, different extension). Each line describes one annotation. |
| `classes.txt` | Text file listing class names, one per line. Line index = class ID. |

---

## Label Format

Each line in a label `.txt` file describes a single annotation. The format depends on the task type:

### Segmentation

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

A polygon defined by `n` normalized coordinate pairs (values in `[0, 1]` relative to image width/height).

### Object Detection

```
<class_id> <x_center> <y_center> <width> <height>
```

A bounding box defined by its center and dimensions, all normalized to `[0, 1]`.

---

## Supported Tasks

| Task | Supported |
|------|:---------:|
| Segmentation | ✅ |
| Object Detection | ✅ |

!!! note
    The YOLO format supports both segmentation and object detection — the label structure determines the task type.

---

## Available Conversions

| Direction | Function |
|-----------|----------|
| YOLO → Label Studio | `yolo_to_ls()` |
| Label Studio → YOLO | `ls_to_yolo()` |
| YOLO → Ultralytics | `yolo_to_ul()` |
| Ultralytics → YOLO | `ul_to_yolo()` |
| BinMask → YOLO | `binmask_to_yolo()` |
<!-- | YOLO → BinMask | `yolo_to_binmask()` | -->

See the [Conversions Usage Guide](../conversions/usage.md) for CLI and API examples.
