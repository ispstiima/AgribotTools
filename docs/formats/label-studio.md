# Label Studio Format

The **Label Studio** (LS) format is the native annotation format used by [Label Studio](https://labelstud.io/), an open-source data labeling tool.

---

## Directory Structure

```
ls_dset/
├── images/
│   ├── img_01.png
│   ├── img_02.png
│   └── ...
├── info.json
└── info.xml
```

### Components

| Component | Description |
|-----------|-------------|
| `images/` | Labelled images in `jpg` or `png` format. |
| `info.json` | JSON file containing image references and their label annotations. |
| `info.xml` | XML configuration file defining the labeling interface. |

---

## XML Configuration Example

The `info.xml` file configures the Label Studio labeling interface:

```xml
<View>
    <!-- View the image to be labelled -->
    <Image name="image" value="$image" />
    <!-- Define the label classes -->
    <Labels name="label" toName="image">
        <Label value="Object" />
    </Labels>

    <!-- Tool for drawing bounding boxes -->
    <RectangleLabels name="bbox" toName="image">
        <Label value="Object" />
    </RectangleLabels>
</View>
```

---

## Supported Tasks

| Task | Supported |
|------|:---------:|
| Segmentation | ✅ |
| Object Detection | ✅ |

!!! note
    The LS format supports both segmentation and object detection. The task type is determined by the label structure in the JSON file.

---

## Large File Import

!!! warning "File size limits"
    Label Studio only supports JSON task files up to **250,000 tasks** or **50 MB**.

    For larger files, use the utility script `scripts/import_tasks_ls.py`, which automatically chunks the file and uploads parts separately to a local Label Studio instance.

---

## Environment Variables

Conversions involving Label Studio require specific environment variables. See the [Getting Started](../getting-started.md#environment-variables) page for configuration details.

---

## Available Conversions

| Direction | Function |
|-----------|----------|
| YOLO → Label Studio | `yolo_to_ls()` |
| Label Studio → YOLO | `yolo_to_ls(reverse)` |
| Label Studio → Ultralytics | `ls_to_ul()` |
| Ultralytics → Label Studio | `ls_to_ul(reverse)` |
| BinMask → Label Studio | `binmask_to_ls()` |

See the [Conversions Usage Guide](../conversions/usage.md) for CLI and API examples.
