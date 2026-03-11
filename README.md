# AgribotTools

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lambdalekter.github.io/AgribotTools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
[![Python ≥3.8](https://img.shields.io/badge/python-≥3.8-blue.svg)](https://www.python.org/downloads/)

**Utilities to convert datasets between Label Studio, YOLO and Ultralytics formats.**

AgribotTools is a Python toolkit developed by [CNR STIIMA](https://www.stiima.cnr.it/) for converting computer-vision annotation datasets across popular formats used in object detection and segmentation tasks.

---

## Features

- **Format conversion** — Seamless conversion between Binary Mask, YOLO, Label Studio, and Ultralytics formats.
- **Validation** — Automatic dataset structure validation before conversion.
<!-- - **Reversible conversions** — Many conversions support reverse mode. -->
- **Gradio GUI** — Web-based graphical interface for point-and-click conversions.
- **CLI scripts** — Ready-to-use command-line scripts for each conversion.
- **Extensible architecture** — Add new formats and conversions via decorators.

---

## Quick Start

```bash
git clone https://github.com/LambdaLekter/AgribotTools.git
cd AgribotTools
uv sync
```

```bash
# Example: Convert YOLO segmentation to Label Studio format
python scripts/yolo_to_ls.py seg /path/to/yolo_dataset --ls_base_name my_ls_output
```

---

## Conversion Workflow

```mermaid
flowchart TB
    subgraph S
        direction LR
        YOLOS(YOLO-S) -- "yolo_to_ls(s)" --> LSS(LS-S)
        LSS -- "yolo_to_ls(s, reverse)" --> YOLOS
        YOLOS -- "yolo_to_ul(s)" --> ULS(UL-S)
        ULS -- "yolo_to_ul(s, reverse)" --> YOLOS
        LSS -- "ls_to_ul(s)" --> ULS
        ULS -- "ls_to_ul(s, reverse)" --> LSS
    end
    subgraph OD
        direction LR
        YOLOD(YOLO-OD) -- "yolo_to_ls(d)" --> LSD(LS-OD)
        LSD -- "yolo_to_ls(d, reverse)" --> YOLOD
        YOLOD -- "yolo_to_ul(d)" --> ULD(UL-OD)
        ULD -- "yolo_to_ul(d, reverse)" --> YOLOD
        LSD -- "ls_to_ul(d)" --> ULD
        ULD -- "ls_to_ul(d, reverse)" --> LSD
    end
    BM(BinMask) -- "binmask_to_yolo(s)" --> YOLOS
    BM -- "binmask_to_yolo(d)" --> YOLOD
    YOLOS -- "binmask_to_yolo(s, reverse)" --> BM
    style BM fill:#03ab
    style S fill:#b03a
    style OD fill:#ab03
```

---

## Documentation

📖 **Full documentation** is available at **[lambdalekter.github.io/AgribotTools](https://lambdalekter.github.io/AgribotTools/)**, including:

- [Getting Started](https://lambdalekter.github.io/AgribotTools/getting-started/) — Installation and environment setup
- [Formats](https://lambdalekter.github.io/AgribotTools/formats/overview/) — Detailed format specifications
- [Conversions](https://lambdalekter.github.io/AgribotTools/conversions/overview/) — Conversion matrix and usage guide
- [GUI](https://lambdalekter.github.io/AgribotTools/gui/usage/) — Gradio interface documentation
- [API Reference](https://lambdalekter.github.io/AgribotTools/api/formats/) — Python API documentation
- [Contributing](https://lambdalekter.github.io/AgribotTools/contributing/) — How to extend AgribotTools

---

## References

* [Ultralytics - Object Detection Datasets Overview](https://docs.ultralytics.com/datasets/detect/)
* [Ultralytics - Instance Segmentation Datasets Overview](https://docs.ultralytics.com/datasets/segment/)
* [Label Studio - Understanding the Label Studio JSON format](https://labelstud.io/blog/understanding-the-label-studio-json-format/#breaking-down-the-label-studio-json-format)
* [Label Studio - Labeling configuration](https://labelstud.io/templates/named_entity#Labeling-Configuration)

---

## License

This project is licensed under the MIT License — see [LICENSE.md](LICENSE.md) for details.
