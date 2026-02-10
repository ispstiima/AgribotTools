"""
Gradio-based GUI for AgribotTools Format Converter.

This module provides a unified graphical interface for converting between
different dataset annotation formats used in computer vision tasks.

Features:
- Dynamic format selection based on registered conversions
- Source format dropdown with all available source formats
- Target format dropdown that updates based on selected source
- Source folder validation before enabling conversion
- Rollback support if conversion fails
- Progress indication during conversion
"""

import gradio as gr
from pathlib import Path
import logging
import sys
import io
import os

# Set a default for Label Studio root if not set (allows GUI to start)
if "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" not in os.environ:
    os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = "/tmp/label_studio_data"
    LS_ENV_WARNING = True
else:
    LS_ENV_WARNING = False

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.cvtoolkit import FormatType, FormatRegistry, ConversionError
from cvtoolkit.formats import TaskType
from src.ui.formats import *
from src.ui.callbacks import *

log = logging.getLogger("ConverterGUI")


def create_gui():
    """Create and configure the Gradio interface."""
    
    image_ext = ".jpg,.png"
    source_path, target_path = None, None
    source_formats = get_source_format_choices()
    initial_source = source_formats[0] if source_formats else None
    initial_targets = get_target_choices(initial_source) if initial_source else []
    initial_target = initial_targets[0] if initial_targets else None
    task_choices = [TaskType.DETECTION.value, TaskType.SEGMENTATION.value]
    initial_task = task_choices[0] if task_choices else None
    
    with gr.Blocks(
        title="AgRibot Format Converter"
    ) as demo:
        gr.Markdown(
            """
            # 🌱 AgRibot Format Converter
            
            > Convert between different dataset annotation formats for computer vision tasks.
            """,
            elem_classes=["main-header"]
        )

        # Info section
        with gr.Accordion("ℹ️ Supported Formats & Structures", open=False):
            gr.Markdown(
                """
                | **Format** | **Description** |
                |--------|-------------|
                | **Binary Mask** | Grayscale mask images (255=foreground, 0=background) |
                | **YOLO (Detection/Segmentation)** | Bounding box / Polygon annotations in normalized coordinates |
                | **Label Studio (Detection/Segmentation)** | JSON format with RLE-encoded masks / Rectangle annotations |
                | **Ultralytics (Detection/Segmentation)** | YOLO format with train/val/test splits |

                ### Directory Structures
                """
            )

            with gr.Row():
                gr.Markdown(
                    """
                    **YOLO / Binary Mask:**
                    ```
                    dataset/
                    ├── images/
                    ├── labels/
                    └── classes.txt
                    ```
                    """
                )
                gr.Markdown(
                    """
                    **Ultralytics:**
                    ```
                    dataset/
                    ├── train/
                    │   ├── images/
                    │   └── labels/
                    ├── val/
                    │   ├── images/
                    │   └── labels/
                    ├── test/
                    │   ├── images/
                    │   └── labels/
                    └── dataset.yaml
                    ```
                    """
                )
                gr.Markdown(
                    """
                    **Label Studio:**
                    ```
                    dataset/
                    ├── images/
                    ├── task.json
                    └── template.label_config.xml
                    ```
                    """
                )
        
        # Show warning if Label Studio environment is not configured
        if LS_ENV_WARNING:
            gr.Markdown(
                """
                > ⚠️ **Warning:** `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` is not set.
                > Label Studio conversions will use `/tmp/label_studio_data`.
                """,
            )
        
        
        with gr.Row():
            source_format = gr.Dropdown(
                choices=source_formats,
                value=initial_source,
                label="Source Format",
                scale=3
            )

            target_format = gr.Dropdown(
                choices=initial_targets,
                value=initial_target,
                label="Target Format",
                scale=3
            )

            task_type = gr.Radio(
                choices=task_choices,
                value=initial_task,
                label="Target Task Type",
                show_label=True,
                interactive=True,
                visible=True,
                scale=3
            )

        with gr.Column():
            with gr.Accordion(label="Source Dataset Path"):
                source_path = gr.FileExplorer(
                    show_label=False,
                    root_dir=".",
                    file_count="single",
                    max_height=230,
                )
        
        validation_status = gr.Markdown(
            "⚠️ Enter a source path to validate.",
            elem_id="validation-status"
        )

        output_name = gr.Textbox(
            label="(Optional)  Target Dataset Name",
            placeholder="Enter a name for the output folder"
        )

        # Split Options (for Ultralytics)
        with gr.Accordion("📊 Split Options", open=False, visible=False) as split_accordion:
            gr.Markdown("Configure train/validation/test splits for Ultralytics format")
            
            with gr.Row():
                train_ratio = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.7, step=0.05,
                    label="Train Ratio",
                )
                val_ratio = gr.Slider(
                    minimum=0.05, maximum=0.5, value=0.2, step=0.05,
                    label="Validation Ratio",
                )
                test_ratio = gr.Slider(
                    minimum=0.0, maximum=0.3, value=0.1, step=0.05,
                    label="Test Ratio",
                )

            include_test = gr.Checkbox(
                label="Include Test Split",
                value=False,
                info="Create a separate test set",
            )
            
            random_seed = gr.Number(
                label="Random Seed",
                value=0,
                precision=0,
                info="0 = random, or set seed for reproducibility",
            )
        
        with gr.Row():
            convert_btn = gr.Button(
                "Convert",
                variant="primary",
                size="lg",
                interactive=False,
            )

            dl_button = gr.DownloadButton(
                label=f"Download {target_format.value} Dataset",
                variant="stop",
                interactive=False,
            )

        output_log = gr.Textbox(
            label="Conversion Log",
            lines=12,
            max_lines=20,
            interactive=False
        )
        
        # Event handlers
        source_format.change(
            fn=update_target_dropdown,
            inputs=[source_format, task_type],
            outputs=[target_format],
        )

        for el in [source_format, target_format]:
            el.change(
                fn=update_task_visibility,
                inputs=[source_format, target_format],
                outputs=[task_type],
            )

        target_format.change(
            fn=update_split_visibility,
            inputs=[target_format],
            outputs=[split_accordion],
        )

        for el in [source_format, source_path, target_format, task_type]:
            el.change(
                fn=update_validation_and_convert,
                inputs=[source_format, source_path, target_format, task_type],
                outputs=[validation_status, convert_btn],
            )
        
        convert_btn.click(
            fn=run_conversion,
            inputs=[
                source_format,
                target_format,
                task_type,
                source_path,
                output_name,
                train_ratio,
                val_ratio,
                test_ratio,
                include_test,
                random_seed,
            ],
            outputs=[output_log, dl_button],
        )

        target_format.change(
            fn=lambda target: gr.update(label=f"Download {target} Dataset"),
            inputs=[target_format],
            outputs=[dl_button]
        )
    
    return demo


# Create the demo instance
demo = create_gui()


if __name__ == "__main__":
    """Entry point for the GUI application."""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

