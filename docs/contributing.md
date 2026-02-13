# Contributing

Thank you for your interest in contributing to AgribotTools! This guide explains how to set up a development environment and add new formats or conversions.

---

## Development Setup

1. **Clone** the repository and install in editable mode:

    ```bash
    git clone https://github.com/LambdaLekter/AgribotTools.git
    cd AgribotTools
    pip install -e .
    ```

2. **Install documentation dependencies** (optional, for building docs locally):

    ```bash
    pip install mkdocs-material mkdocstrings[python]
    ```

3. **Run the docs locally**:

    ```bash
    mkdocs serve
    ```

---

## Project Structure

```
AgribotTools/
├── src/
│   ├── cvtoolkit/
│   │   ├── formats/        # Format definitions and validators
│   │   │   ├── format.py   # Base classes: Format, FormatType, FormatRegistry
│   │   │   ├── binmask.py  # Binary Mask format
│   │   │   ├── yolo.py     # YOLO format
│   │   │   ├── ls.py       # Label Studio format
│   │   │   └── ul.py       # Ultralytics format
│   │   ├── conversions/    # Conversion implementations
│   │   │   ├── conversion.py  # Base: Conversion, ReversibleConversion
│   │   │   ├── yolo_to_ls.py
│   │   │   ├── yolo_to_ul.py
│   │   │   └── ...
│   │   ├── rle.py          # Run-length encoding utilities
│   │   ├── mask.py         # Mask/contour utilities
│   │   └── colors.py       # Color palette for visualizations
│   ├── ui/                 # Gradio GUI helpers
│   └── file_utils.py       # File copy utilities
├── scripts/                # CLI entry points
├── docs/                   # MkDocs documentation (this site)
├── gui.py                  # Gradio GUI entry point
├── mkdocs.yml              # MkDocs configuration
└── setup.cfg               # Package configuration
```

---

## Adding a New Format

Formats are registered using the `@register_format` decorator.

### Step 1: Create a format module

Create a new file in `src/cvtoolkit/formats/`, e.g., `coco.py`:

```python
from pathlib import Path
from typing import Tuple
from cvtoolkit.formats.format import Format, FormatType, register_format


@register_format(FormatType.COCO)  # Add COCO to FormatType enum first
class Coco(Format):
    """Validator for the COCO annotation format."""
    
    display_name = "COCO"
    
    def validate_structure(self) -> Tuple[bool, str]:
        """Check that the directory has the expected COCO structure."""
        annotations = self.path / "annotations"
        images = self.path / "images"
        
        if not annotations.exists():
            return False, "Missing 'annotations/' directory."
        if not images.exists():
            return False, "Missing 'images/' directory."
        
        return True, "Valid COCO dataset."
```

### Step 2: Add the FormatType enum value

In `src/cvtoolkit/formats/format.py`, add the new value to `FormatType`:

```python
class FormatType(Enum):
    BINMASK = auto()
    YOLO = auto()
    LABEL_STUDIO = auto()
    ULTRALYTICS = auto()
    COCO = auto()  # New
```

### Step 3: Register on import

In `src/cvtoolkit/__init__.py`, add the import to `initialize_registry()`:

```python
def initialize_registry():
    from cvtoolkit.formats import binmask, ls, ul, yolo, coco  # add coco
    ...
```

---

## Adding a New Conversion

Conversions are registered using the `@register_conversion` decorator.

### Step 1: Create a conversion module

Create a new file in `src/cvtoolkit/conversions/`, e.g., `yolo_to_coco.py`:

```python
from pathlib import Path
from cvtoolkit.conversions.conversion import (
    Conversion, 
    ReversibleConversion, 
    register_conversion
)
from cvtoolkit.formats.format import FormatType


@register_conversion(FormatType.YOLO, FormatType.COCO)
class YoloToCoco(Conversion):
    """Convert YOLO format to COCO format."""
    
    def convert(self, **kwargs) -> Path:
        """Perform the YOLO → COCO conversion."""
        # Create output directory
        output_dir = self.target_path
        output_dir.mkdir(parents=True, exist_ok=True)
        self._track_path(output_dir)
        
        # ... conversion logic ...
        
        self._report_progress(1.0, "Conversion complete.")
        return output_dir
```

### Step 2: Register on import

In `src/cvtoolkit/__init__.py`, add the import:

```python
def initialize_registry():
    ...
    from cvtoolkit.conversions import yolo_to_coco  # add new conversion
```

### Step 3: Create a CLI script (optional)

Create `scripts/yolo_to_coco.py` following the pattern of existing scripts.

---

## Coding Conventions

- **Docstrings**: Use Google-style docstrings. These are automatically picked up by `mkdocstrings`.
- **Type hints**: All public functions should have type annotations.
- **Logging**: Use the module-level logger (`log = logging.getLogger("...")`) instead of `print()`.
- **Progress tracking**: Call `self._report_progress(fraction, message)` during long operations.
- **Rollback support**: Call `self._track_path(path)` for any files/directories your conversion creates.
