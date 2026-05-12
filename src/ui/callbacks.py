import logging
import shutil
import gradio as gr
from pathlib import Path
from tempfile import TemporaryDirectory
from cvtoolkit import FormatType, FormatRegistry, ConversionError
from cvtoolkit.formats import TaskType
from ui.formats import get_format_type_by_name, get_target_choices, validate_source_folder

log = logging.getLogger("ConverterGUI")


def update_target_dropdown(source_format: str, source_task: str):
    """Update target dropdown choices when source changes."""
    targets = get_target_choices(source_format)
    default_value = targets[0] if targets else None
    return gr.update(choices=targets, value=default_value)


def update_validation_and_convert(source_format: str, source_path: str, target_format: str, task_type: str, path_in_yaml: str):
    """Validate source and update convert button state."""
    message, source_valid = validate_source_folder(source_format, source_path)

    if not source_valid:
        return message, gr.update(interactive=False)
    
    if not target_format:
        return "⚠️ Please select a target format.", gr.update(interactive=False)
    
    target_type = get_format_type_by_name(target_format)

    if target_type == FormatType.ULTRALYTICS and not path_in_yaml.strip():
        return "⚠️ Please fill the Dataset Path in the Ultralytics Options.", gr.update(interactive=False)
    
    return message, gr.update(interactive=True)


def update_split_visibility(target_format: str):
    """Show/hide split options based on target format."""
    show_splits = target_format and "ultralytics" in target_format.lower()
    return gr.update(visible=show_splits)


def update_split_validation(train_ratio, val_ratio, test_ratio, include_test):
    val_msg = "✅ The splits will be created as specified."
    total_ratio = train_ratio + val_ratio + test_ratio

    if train_ratio <= 0 or val_ratio <= 0:
        val_msg = f"❌ The training and validation ratios must be greater than 0."
    elif abs(total_ratio - 1.0) > 1e-6:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        val_msg = f"⚠️ Split ratios do not sum to 1. They will be normalized to: {train_ratio:.2f} / {val_ratio:.2f}"

        if include_test:
            val_msg +=  f" / {test_ratio:.2f}"
    
    return gr.update(value=val_msg)


def update_task_visibility(source_format: str, target_format: str):
    """Show/hide task options based on source format."""
    conversions_with_task = [
        (
            FormatRegistry.get_display_name(f1),
            FormatRegistry.get_display_name(f2)
        ) for f1, f2 in [
            (FormatType.BINMASK, FormatType.YOLO),
            (FormatType.BINMASK, FormatType.LABEL_STUDIO),
            (FormatType.LABEL_STUDIO, FormatType.YOLO),
            (FormatType.YOLO, FormatType.LABEL_STUDIO),
            (FormatType.LABEL_STUDIO, FormatType.ULTRALYTICS),
            (FormatType.ULTRALYTICS, FormatType.LABEL_STUDIO),
        ]
    ]
    
    tasks_visible = True
    if (source_format, target_format) not in conversions_with_task:
        tasks_visible = "hidden"
    return gr.update(visible=tasks_visible)


def run_conversion(
    source_format: str,
    target_format: str,
    task_type_str: str,
    source_path: str,
    output_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    include_test: bool,
    random_seed: int,
    path_in_yaml: str,
    image_ext: str = ".jpg,.png",
    progress: gr.Progress = gr.Progress(),
):
    """
    Execute the selected conversion operation (generator for live log updates).
    
    Yields tuples of (progress_text, output_log, dl_button) as the conversion
    progresses, so the UI updates in real time.
    """
    success = False
    result_messages = []
    
    def _emit(msg: str):
        """Append a message to the log and return the current state to yield."""
        result_messages.append(msg)
        return (
            gr.update(),
            "\n".join(result_messages),
            gr.update(),
            gr.update()
        )
    
    # Convert string to TaskType enum
    task_type = TaskType(task_type_str)
    
    # Immediately disable the download button when conversion starts
    yield (
        gr.update(placeholder="Converting..."),
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    
    try:
        if not source_path:
            yield _emit("❌ Error: Source path is required.")
            return
        
        source_path = Path(source_path)
        
        progress(0, desc="Initializing...")
        
        # Get format types
        source_type = get_format_type_by_name(source_format)
        target_type = get_format_type_by_name(target_format)
        
        if source_type is None:
            yield _emit(f"❌ Error: Unknown source format: {source_format}")
            return
        
        if target_type is None:
            yield _emit(f"❌ Error: Unknown target format: {target_format}")
            return
        
        # Get conversion class
        conversion_class = FormatRegistry.get_conversion_class(source_type, target_type)
        if conversion_class is None:
            yield _emit(f"❌ Error: No conversion available from {source_format} to {target_format}")
            return
        
        if not output_name:
            output_name = source_path.name + "_" + target_format.lower()

        # Clean the out/ folder before starting a new conversion
        out_path = Path("out")
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(exist_ok=True)
        target_path = out_path / output_name
        
        yield _emit(f"🔄 Starting conversion: {source_format} → {target_format}")
        yield _emit(f"📂 Source: {source_path}")
        yield _emit(f"📁 Target: {target_path}")
        yield _emit(f"🎯 Task Type: {task_type_str}")
        
        # Create converter
        converter = conversion_class(source_path, target_path, task_type)
        
        # Set up progress callback to update Gradio progress bar
        def progress_callback(pct: float, message: str):
            progress(pct, desc=message)
        
        converter.set_progress_callback(progress_callback)
        
        # Build kwargs based on target format
        kwargs = {}
        
        # Add split options for Ultralytics targets
        if target_type == FormatType.ULTRALYTICS:
            split_ratios = (train_ratio, val_ratio, test_ratio) if include_test else (train_ratio, val_ratio)

            if path_in_yaml.strip():
                kwargs["path_in_yaml"] = path_in_yaml.strip()
            else:
                yield _emit(f"❌ Error: Path in YAML is required for Label Studio to Ultralytics conversion.")
                return
            
            kwargs.update({
                "split_ratios": split_ratios,
                "include_test_split": include_test,
                "image_ext": image_ext,
                "random_seed": random_seed if random_seed > 0 else None,
            })
        
        # Run conversion with rollback support
        result = converter.run(**kwargs)
        
        progress(1.0, desc="Complete!")

        yield (
            gr.update(placeholder="Compressing output dataset..."),
            gr.update(),
            gr.update(),
            gr.update(),
        )
        
        yield _emit("✅ Conversion completed successfully!")
        yield _emit(f"📁 Output saved to: {result}")

        # Zip the output directory for download
        zip_path = shutil.make_archive(
            base_name=str(target_path),
            format="zip",
            root_dir=target_path.parent,
            base_dir=target_path.name,
        )
        yield _emit(f"📦 Download ready: {zip_path}")

        success = True
        
    except ConversionError as e:
        yield _emit(f"❌ Conversion failed: {str(e)}")
        yield _emit("🔄 Changes have been rolled back.")
        log.exception("Conversion error")
    
    except Exception as e:
        yield _emit(f"❌ Unexpected error: {str(e)}")
        log.exception("Unexpected error")
    
    if success:
        dl_update = gr.update(interactive=True, value=zip_path)
        convert_update = gr.update()
    else:
        dl_update = gr.update()
        convert_update = gr.update(interactive=True)
    
    yield (
        gr.update(placeholder="Waiting for the conversion to start..."),
        "\n".join(result_messages),
        dl_update,
        convert_update,
    )


def update_source_text(path):
    if path is None:
        text = "Invalid path"
    elif not Path(path).is_dir():
        text = f"❌ ERROR: The selected source path is not a directory"
    else:
        text = f"{str(Path(path))}"
    
    return gr.update(value=text)

