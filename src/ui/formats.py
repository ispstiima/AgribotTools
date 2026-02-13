from pathlib import Path
from cvtoolkit import FormatRegistry, FormatType
from ui.utils import get_folder_path


def get_source_format_choices() -> list:
    """Get list of source format display names for dropdown."""
    choices = FormatRegistry.get_format_choices()
    return [name for name, _ in choices]


def get_format_type_by_name(display_name: str) -> FormatType | None:
    """Get FormatType enum by display name."""
    for fmt in FormatType:
        if FormatRegistry.get_display_name(fmt) == display_name:
            return fmt
    return None


def get_target_choices(source_name: str) -> list:
    """Get list of valid target format names for a given source."""
    source_type = get_format_type_by_name(source_name)
    if source_type is None:
        return []
    
    targets = FormatRegistry.get_supported_targets(source_type)
    return [FormatRegistry.get_display_name(t) for t in targets]


def validate_source_folder(source_format: str, source_path: str) -> tuple:
    """
    Validate that the source folder matches the expected format structure.
    
    Returns:
        Tuple of (message: str, source_valid: bool)
    """
    if source_path is None:
        return "❌ Path does not exist.", False
    
    source_path = Path(source_path)
    
    if not source_path.is_dir():
        return "❌ Path is not a directory.", False
    
    source_type = get_format_type_by_name(source_format)
    if source_type is None:
        return "❌ Unknown source format.", False
    
    format_class = FormatRegistry.get_format_class(source_type)
    if format_class is None:
        return "❌ No validator for format.", False
    
    validator = format_class(source_path)
    is_valid, message = validator.validate()
    
    if is_valid:
        return f"✅ Valid {source_format} dataset: {source_path.resolve()}", True
    else:
        return f"❌ Validation failed: {message}", False

