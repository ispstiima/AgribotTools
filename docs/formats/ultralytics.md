# Ultralytics Format

The **Ultralytics** (UL) format is the dataset structure expected by the [Ultralytics](https://docs.ultralytics.com/) training library. It organizes data into train/val/test splits with a YAML configuration file.

---

## Directory Structure

```
ul_dset/
├── train/
│   ├── images/
│   │   ├── img_01.png
│   │   └── ...
│   └── labels/
│       ├── img_01.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── img_01.png
│   │   └── ...
│   └── labels/
│       ├── img_01.txt
│       └── ...
├── test/          # optional
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── ul_dset.yaml
```

### Components

| Component | Description |
|-----------|-------------|
| `train/` | Training split with `images/` and `labels/` subfolders. |
| `val/` | Validation split with the same structure. |
| `test/` | *(Optional)* Test split with the same structure. |
| `ul_dset.yaml` | YAML configuration file (named after the dataset). |

!!! note
    The `train`, `val`, and `test` subfolders share the same internal structure. Labels use the YOLO text format.

---

## YAML Configuration

The YAML file configures paths and class names for the Ultralytics training pipeline:

```yaml
# Dataset name
path: /path/to/dataset              # Absolute path to dataset root
train: /train/images                 # Training images (relative to path)
val: /val/images                     # Validation images (relative to path)
test: /test/images                   # Test images (relative to path, optional)

# Class names
names:
    0: first_class
    1: second_class
    2: ...
```

---

## Supported Tasks

| Task | Supported |
|------|:---------:|
| Segmentation | ✅ |
| Object Detection | ✅ |

!!! note
    The UL format supports both segmentation and object detection. The task type is determined by the label structure (same as YOLO labels).

---

## Available Conversions

| Direction | Function |
|-----------|----------|
| YOLO → Ultralytics | `yolo_to_ul()` |
| Ultralytics → YOLO | `ul_to_yolo()` |
| Label Studio → Ultralytics | `ls_to_ul()` |
| Ultralytics → Label Studio | `ul_to_ls()` |

See the [Conversions Usage Guide](../conversions/usage.md) for CLI and API examples.
