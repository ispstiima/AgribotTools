"""
CVToolkit - Computer Vision format conversion toolkit.

This package provides:
- Format definitions and validation (YOLO, Label Studio, Ultralytics, Binary Mask)
- Conversions between formats
- GUI support with format registry

Usage:
    from cvtoolkit.formats.format import FormatType, FormatRegistry
    from cvtoolkit.conversions.conversion import Conversion
    
    # Get available source formats
    sources = FormatRegistry.get_all_source_formats()
    
    # Get valid targets for a source
    targets = FormatRegistry.get_supported_targets(FormatType.YOLO_SEG)
    
    # Get conversion class
    converter_class = FormatRegistry.get_conversion_class(
        FormatType.YOLO_SEG, 
        FormatType.LABEL_STUDIO_SEG
    )
    
    # Run conversion
    converter = converter_class(source_path, target_path)
    result = converter.run()
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from cvtoolkit.formats import FormatType, FormatRegistry


log = logging.getLogger("cvtoolkit")


# Build SUPPORTED_CONVERSIONS from registry
def get_supported_conversions() -> dict:
    """Get supported conversions from the registry."""
    conversions = {}
    for source in FormatType:
        targets = FormatRegistry.get_supported_targets(source)
        if targets:
            conversions[source] = targets
    return conversions


SUPPORTED_CONVERSIONS = get_supported_conversions()


def load_env(dotenv_path: str | Path | None = None, raise_on_missing: bool = True) -> dict:
    """
    Load environment variables from a .env file.
    
    Args:
        dotenv_path: Path to .env file. If None, looks in repository root.
        raise_on_missing: If True, raise error when required vars are missing.
    
    Returns:
        Dictionary with loaded environment values.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"

    try:
        load_dotenv(dotenv_path)
    except Exception as e:
        log.debug("Ignored error loading dotenv %s: %s", dotenv_path, e)

    ls_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", None)
    if ls_root is None:
        msg = (
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT environment variable not set. "
            "Please set it to the root directory of your Label Studio installation."
        )
        if raise_on_missing:
            raise EnvironmentError(msg)
        else:
            log.warning(msg)
            ls_root = "/tmp/label_studio_data"

    return {
        "LS_ROOT_PATH": Path(ls_root)
    }


def initialize_registry():
    """
    Initialize the format and conversion registry by importing all modules.
    This triggers the @register_format and @register_conversion decorators.
    """
    from cvtoolkit.formats import binmask, ls, ul, yolo
    from cvtoolkit.conversions import (
        binmask_to_ls,
        binmask_to_yolo,
        ls_to_yolo,
        ul_to_yolo,
        yolo_to_ls,
        yolo_to_ul,
    )


initialize_registry()


from cvtoolkit.formats.format import FormatType, FormatRegistry, Format
from cvtoolkit.conversions.conversion import Conversion, ConversionError


__all__ = [
    "FormatType",
    "FormatRegistry", 
    "Format",
    "Conversion",
    "ConversionError",
    "load_env",
    "initialize_registry",
]
