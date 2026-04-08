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
import os

# Set a default for Label Studio root if not set (allows GUI to start)
# if "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT" not in os.environ:
#     os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = "/tmp/label_studio_data"
#     LS_ENV_WARNING = True
# else:
#     LS_ENV_WARNING = False

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cvtoolkit.formats import TaskType
from ui.formats import *
from ui.callbacks import *

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
    
    with gr.Blocks(title="AgRibot Format Converter") as demo:
        gr.Markdown(
            """
            # 🌱 AgRibot Format Converter
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                source_format = gr.Dropdown(
                    choices=source_formats,
                    value=initial_source,
                    label="Source Format",
                )

                target_format = gr.Dropdown(
                    choices=initial_targets,
                    value=initial_target,
                    label="Target Format",
                )

                task_type = gr.Radio(
                    choices=task_choices,
                    value=initial_task,
                    label="Target Task Type",
                    show_label=True,
                    interactive=True,
                    visible=True,
                )
            
            with gr.Column(scale=3):
                source_path = gr.FileExplorer(
                    label="Source Path Dataset",
                    root_dir="data/" if Path("data/").exists() else "/",
                    file_count="single",
                    max_height=262,
                )

                source_validation = gr.Markdown(
                    "⚠️ Select the source folder path.",
                    elem_id="validation-status"
                )

        with gr.Accordion("🛠️ Ultralytics Options", open=True, visible=False) as split_options:
            with gr.Row():
                with gr.Column(scale=2):
                    path_in_yaml = gr.Textbox(
                        label="Dataset Path",
                        placeholder="Enter a path",
                        info="Path to the folder containing the dataset as specified in the configuration file",
                        max_lines=2,
                    )

                    random_seed = gr.Number(
                        label="Random Seed",
                        value=0,
                        precision=0,
                        info="0 = random, or set seed for reproducibility",
                    )
                
                with gr.Column(scale=3):
                    include_test = gr.Checkbox(
                        label="Include Test Split",
                        value=False,
                    )
                    train_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                        label="Train Ratio",
                    )
                    val_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                        label="Validation Ratio",
                    )
                    test_ratio = gr.Slider(
                        visible="hidden",
                        minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                        label="Test Ratio",
                    )
            
                    splits_validation = gr.Markdown("✅ The splits will be created.")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=2):
                output_name = gr.Textbox(
                    label="(Optional) Target Dataset Name",
                    placeholder="Enter a name for the output folder"
                )

                with gr.Row():
                    convert_btn = gr.Button(
                        "Convert",
                        icon="res/sync_24dp.svg",
                        variant="primary",
                        size="lg",
                        interactive=False,
                    )

                    stop_button = gr.Button(
                        "Stop",
                        icon="res/stop_24dp.svg",
                        variant="stop",
                        size="lg",
                        interactive=False,
                    )

                progress_text = gr.Textbox(
                    container=False,
                    placeholder="Waiting for the conversion to start...",
                    show_label=False,
                    interactive=False,
                    lines=3
                )

                dl_button = gr.DownloadButton(
                    label=f"Download",
                    icon="res/download_24dp.svg",
                    variant="primary",
                    size="lg",
                    interactive=False,
                )

            output_log = gr.Textbox(
                label="Conversion Log",
                lines=12,
                max_lines=20,
                interactive=False,
                scale=3,
                elem_id="conversion-log"
            )
        
        # Info section
        with gr.Accordion("Formats Description", open=False):
            gr.Markdown(
                """
                ## Supported Formats
                | **FORMAT** | **DESCRIPTION** |
                |--------|-------------|
                | **Binary Mask** | Binary mask images (255=foreground, 0=background) |
                | **YOLO (Detection/Segmentation)** | Bounding box / Polygon annotations in normalized coordinates |
                | **Label Studio (Detection/Segmentation)** | JSON format with RLE-encoded masks / Rectangle annotations |
                | **Ultralytics (Detection/Segmentation)** | YOLO format with train/val/test splits |

                ## Directory Structures
                """
            )

            with gr.Row():
                gr.Markdown(
                    """
                    ### **YOLO / Binary Mask**
                    ```
                    dataset/
                    ├── images/
                    ├── labels/
                    └── classes.txt
                    """ + '\n'*7 + """
                    ```
                    """, container=True
                )
                gr.Markdown(
                    f"""
                    ### **Label Studio**
                    ```
                    dataset/
                    ├── images/
                    ├── task.json
                    └── template.label_config.xml
                    """ + '\n'*7 + """
                    ```
                    """, container=True
                )
                gr.Markdown(
                    """
                    ### **Ultralytics**
                    ```
                    dataset/
                    ├── train/
                    │   ├── images/
                    │   └── labels/
                    ├── val/
                    │   ├── images/
                    │   └── labels/
                    ├── test/         # Optional
                    │   ├── images/
                    │   └── labels/
                    └── dataset.yaml
                    ```
                    """, container=True
                )
        
        # ========================== Event handlers ==========================

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
            outputs=[split_options],
        )

        for el in [source_format, source_path, target_format, task_type, path_in_yaml]:
            el.change(
                fn=update_validation_and_convert,
                inputs=[source_format, source_path, target_format, task_type, path_in_yaml],
                outputs=[source_validation, convert_btn],
            )
        
        include_test.select(
            fn=lambda selected: gr.update(visible=True if selected else "hidden", value=0.0),
            inputs=[include_test],
            outputs=[test_ratio],
        )

        for el in [train_ratio, val_ratio, test_ratio, include_test]:
            el.change(
                fn=update_split_validation,
                inputs=[train_ratio, val_ratio, test_ratio, include_test],
                outputs=[splits_validation],
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
                path_in_yaml
            ],
            outputs=[progress_text, output_log, dl_button, convert_btn],
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
        css="""
            #conversion-log div[data-testid=\"status-tracker\"] {
                display: none !important;
            }

            .code_wrap > .copy_code_button {
                display: none !important;
            }
        """
    )

