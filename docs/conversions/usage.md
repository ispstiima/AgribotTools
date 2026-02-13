# Conversions Usage

AgribotTools provides two ways to run format conversions: **CLI scripts** and the **Python API**.

---

## CLI Scripts

Ready-to-use scripts are located in the `scripts/` directory. Each script handles a specific conversion and accepts command-line arguments.

### Available Scripts

| Script | Conversion |
|--------|-----------|
| `binmask_to_yolo.py` | BinMask → YOLO |
| `yolo_to_binmask.py` | YOLO → BinMask |
| `binmask_to_ls.py` | BinMask → Label Studio |
| `yolo_to_ls.py` | YOLO → Label Studio |
| `ls_to_yolo.py` | Label Studio → YOLO |
| `yolo_to_ul.py` | YOLO → Ultralytics |
| `ul_to_yolo.py` | Ultralytics → YOLO |
| `ls_to_ul.py` | Label Studio → Ultralytics |
| `ul_to_ls.py` | Ultralytics → Label Studio |
| `seg_ls_to_bbox_ls.py` | Label Studio segmentation → Label Studio detection |
| `import_tasks_ls.py` | Bulk import tasks into Label Studio |

### Example: YOLO → Label Studio

```bash
python scripts/yolo_to_ls.py seg /path/to/yolo_dataset --ls_base_name my_output
```

### Example: YOLO → Ultralytics

```bash
python scripts/yolo_to_ul.py /path/to/yolo_dataset \
    --ul_path /path/to/output \
    --split_ratios 0.8 0.1 0.1 \
    --include_test_split \
    --image_ext .jpg,.png \
    --random_seed 42
```

### Getting Help

To see all available options for any script:

```bash
python scripts/<script_name>.py -h
```

---

## Python API

For programmatic usage, you can use the conversion classes directly.

### Using the FormatRegistry (Recommended)

The `FormatRegistry` provides a dynamic lookup of available conversions:

```python
from cvtoolkit.formats.format import FormatType, FormatRegistry

# Get all available source formats
sources = FormatRegistry.get_all_source_formats()

# Get valid target formats for a source
targets = FormatRegistry.get_supported_targets(FormatType.YOLO)

# Get the conversion class
converter_class = FormatRegistry.get_conversion_class(
    FormatType.YOLO, 
    FormatType.LABEL_STUDIO
)

# Run the conversion
converter = converter_class(
    source_path="/path/to/yolo_dataset",
    target_path="/path/to/output"
)
result = converter.run()
print(f"Converted to: {result}")
```

### Using Conversion Classes Directly

You can also import and use specific conversion classes:

```python
from cvtoolkit.conversions.yolo_to_ul import YoloToUltralytics
from cvtoolkit.formats import TaskType

converter = YoloToUltralytics(
    source_path="/path/to/yolo_dataset",
    target_path="/path/to/output",
    task_type=TaskType.SEGMENTATION
)

result = converter.run(
    split_ratios=(0.8, 0.1, 0.1),
    include_test_split=True,
    image_ext=".jpg,.png",
    random_seed=42,
)
```

### Reverse Conversions

For reversible conversions, use `run_reverse()`:

```python
from cvtoolkit.conversions.yolo_to_ul import YoloToUltralytics
from cvtoolkit.formats import TaskType

# Note: source/target are swapped for reverse
converter = YoloToUltralytics(
    source_path="/path/to/ul_dataset",
    target_path="/path/to/yolo_output",
    task_type=TaskType.GENERIC
)
result = converter.run_reverse()
```

### Progress Tracking

Conversions support progress callbacks for integration with GUIs or logging:

```python
def on_progress(progress: float, message: str):
    print(f"[{progress:.0%}] {message}")

converter.set_progress_callback(on_progress)
converter.run()
```

---

## Error Handling

Conversions raise `ConversionError` on failure:

```python
from cvtoolkit.conversions.conversion import ConversionError

try:
    result = converter.run()
except ConversionError as e:
    print(f"Conversion failed: {e}")
    # Partial files are automatically rolled back
```
