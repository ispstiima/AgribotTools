# AgribotTools

**Utilities to convert datasets between Label Studio, YOLO and Ultralytics formats.**

AgribotTools is a Python toolkit developed by [CNR STIIMA](https://www.stiima.cnr.it/) for converting computer-vision annotation datasets across popular formats used in object detection and segmentation tasks.

---

## Features

- **Format conversion** — Convert seamlessly between Binary Mask, YOLO, Label Studio, and Ultralytics formats.
- :**Validation** — Automatically validate dataset structure before conversion.
<!-- - **Reversible conversions** — Many conversions support reverse mode out of the box. -->
- **Gradio GUI** — A web-based graphical interface for point-and-click conversions.
- **CLI scripts** — Ready-to-use command-line scripts for each conversion.
- **Extensible architecture** — Add new formats and conversions via simple decorators.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/ispstiima/AgribotTools.git
cd AgribotTools
uv sync

# Convert YOLO segmentation to Label Studio
python scripts/yolo_to_ls.py seg /path/to/yolo_dataset --ls_base_name my_ls_output
```

For detailed setup instructions, see [Getting Started](getting-started.md).

---

## Supported Formats

| Format | Description |
|--------|-------------|
| **Binary Mask** | Binary segmentation masks |
| **YOLO** | Standard YOLO annotation format (segmentation & detection) |
| **Label Studio** | JSON + XML format used by Label Studio |
| **Ultralytics** | Train/val/test split format used by Ultralytics |

See the full [Formats Overview](formats/overview.md) and [Conversion Matrix](conversions/overview.md).

---

## Project Links

- :fontawesome-brands-github: [GitHub Repository](https://github.com/LambdaLekter/AgribotTools)
- :material-license: [License (MIT)](https://github.com/LambdaLekter/AgribotTools/blob/main/LICENSE.md)
