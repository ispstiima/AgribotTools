# Formats Overview

AgribotTools supports four annotation formats used in computer vision for **object detection** and **segmentation** tasks.

---

## Acronyms and Definitions

### Tools

| Acronym | Explanation |
|---------|-------------|
| **LS** | **Label Studio** — the software used for labeling. |
| **UL** | **Ultralytics** — target library for object detection and segmentation. |
| **BinMask** | **Binary mask** — binary image where `1` represents a foreground pixel, `0` represents background. |
| **SegMask** | **Segmentation mask** — mask used to highlight an object of interest. |
| **Bbox** | **Bounding box** — rectangle used to highlight an object of interest. |

### Task Types

| Acronym | Definition | Description |
|---------|------------|-------------|
| **S** | Segmentation | A segmentation task. |
| **OD** | Object Detection | An object detection task. |

### Format Identifiers

| Acronym | Definition | Description |
|---------|------------|-------------|
| **BinMask** | Binary Mask | Segmentation mask for several classes in a binary format. |
| **YOLO** | YOLO format | Standard YOLOv1-v3 annotation format. |
| **LS** | Label Studio | Format used by Label Studio. |
| **UL** | Ultralytics | Format used by Ultralytics. |

---

## Supported Formats

| Format | Segmentation | Object Detection | Details |
|--------|:---:|:---:|---------|
| [Binary Mask](binmask.md) | :check: | :check: | Binary PNG masks |
| [YOLO](yolo.md) | :check: | :check: | Polygon or bbox text annotations |
| [Label Studio](label-studio.md) | :check: | :check: | JSON + XML configuration |
| [Ultralytics](ultralytics.md) | :check: | :check: | Train/val/test split with YAML config |

---

## Format Selection Guide

- **Starting from scratch?** Use the **YOLO** format — it is the central hub for most conversions.
- **Need to label data?** Convert to **Label Studio** format and use the Label Studio UI.
- **Ready to train?** Convert to **Ultralytics** format for direct use with `ultralytics` training pipelines.
- **Have binary masks from another tool?** Start with **Binary Mask** format and convert as needed.
