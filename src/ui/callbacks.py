import os
import logging
import gradio as gr
from pathlib import Path
from src.cvtoolkit import FormatType, FormatRegistry, ConversionError
from cvtoolkit.formats import TaskType
from src.ui.formats import get_format_type_by_name, get_target_choices, validate_source_folder

log = logging.getLogger("ConverterGUI")


def update_target_dropdown(source_format: str, source_task: str):
    """Update target dropdown choices when source changes."""
    targets = get_target_choices(source_format)
    default_value = targets[0] if targets else None
    return gr.update(choices=targets, value=default_value)


def update_validation_and_convert(source_format: str, source_path: str, target_format: str, task_type: str):
    """Validate source and update convert button state."""
    message, source_valid = validate_source_folder(source_format, source_path)
    
    if source_valid and not target_format:
        source_valid = False
        message = "⚠️ Please select a target format."
    
    return message, gr.update(interactive=source_valid)


def update_split_visibility(target_format: str):
    """Show/hide split options based on target format."""
    show_splits = target_format and "ultralytics" in target_format.lower()
    return gr.update(visible=show_splits)


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
    target_path: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    include_test: bool,
    random_seed: int,
    image_ext: str = ".jpg,.png",
    progress: gr.Progress = gr.Progress(),
) -> str:
    """
    Execute the selected conversion operation.
    
    Args:
        source_format: Display name of source format
        target_format: Display name of target format
        source_path: list of files from the input dataset
        target_path: Path for the output dataset
        train_ratio: Training set ratio for splits
        val_ratio: Validation set ratio for splits
        test_ratio: Test set ratio for splits
        include_test: Whether to include a test split
        random_seed: Random seed for reproducible splits
        image_ext: Comma-separated list of image extensions
        progress: Gradio progress tracker
    
    Returns:
        A string containing the conversion result or error message.
    """
    result_messages = []
    
    # Convert string to TaskType enum
    task_type = TaskType(task_type_str)
    
    try:
        if not source_path:
            return "❌ Error: Source path is required."
        
        source_path = Path(source_path)
        
        progress(0, desc="Initializing...")
        
        # Get format types
        source_type = get_format_type_by_name(source_format)
        target_type = get_format_type_by_name(target_format)
        
        if source_type is None:
            return f"❌ Error: Unknown source format: {source_format}"
        
        if target_type is None:
            return f"❌ Error: Unknown target format: {target_format}"
        
        # Get conversion class
        conversion_class = FormatRegistry.get_conversion_class(source_type, target_type)
        if conversion_class is None:
            return f"❌ Error: No conversion available from {source_format} to {target_format}"
        
        # Determine output path
        if not target_path:
            source_suffix = source_format.lower().replace(" ", "_")
            target_suffix = target_format.lower().replace(" ", "_")
            output_name = f"{source_path.name}_{target_suffix}"

            if target_type == FormatType.LABEL_STUDIO:
                ls_base_path = os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", source_path.parent)
                target_path = Path(ls_base_path) / output_name
            else:
                target_path = source_path.parent / output_name
        else:
            target_path = Path(target_path)
        
        result_messages.append(f"🔄 Starting conversion: {source_format} → {target_format}")
        result_messages.append(f"📂 Source: {source_path}")
        result_messages.append(f"📁 Target: {target_path}")
        result_messages.append(f"🎯 Task Type: {task_type_str} → {task_type}")
        
        # Create converter
        converter = conversion_class(source_path, target_path, task_type)
        
        # Set up progress callback to update Gradio progress bar
        def progress_callback(pct: float, message: str):
            progress(pct, desc=message)
        
        converter.set_progress_callback(progress_callback)
        
        # Build kwargs based on target format
        kwargs = {}
        
        # Add split options for Ultralytics targets
        if "ultralytics" in target_format.lower():
            split_ratios = (train_ratio, val_ratio, test_ratio) if include_test else (train_ratio, val_ratio)
            kwargs.update({
                "split_ratios": split_ratios,
                "include_test_split": include_test,
                "image_ext": image_ext,
                "random_seed": random_seed if random_seed > 0 else None,
            })
        
        # Run conversion with rollback support
        result = converter.run(**kwargs)
        
        progress(1.0, desc="Complete!")
        
        result_messages.append("✅ Conversion completed successfully!")
        result_messages.append(f"📁 Output saved to: {result}")
        
    except ConversionError as e:
        result_messages.append(f"❌ Conversion failed: {str(e)}")
        result_messages.append("🔄 Changes have been rolled back.")
        log.exception("Conversion error")
    
    except Exception as e:
        result_messages.append(f"❌ Unexpected error: {str(e)}")
        log.exception("Unexpected error")
    
    return "\n".join(result_messages)

