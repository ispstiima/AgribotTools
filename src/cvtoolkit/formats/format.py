"""
Format base class and registry for dataset format validation and conversion.

This module provides the foundational architecture for:
- Defining dataset formats (YOLO, Label Studio, Ultralytics, BinMask)
- Validating dataset structure
- Tracking supported conversions between formats
"""

import abc
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Tuple, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cvtoolkit.conversions.conversion import Conversion


class FormatType(Enum):
    """Enumeration of supported annotation format types."""
    BINMASK = auto()
    YOLO = auto()
    LABEL_STUDIO = auto()
    ULTRALYTICS = auto()


class FormatRegistry:
    """
    Central registry for format classes and their supported conversions.
    
    This registry enables:
    - Dynamic discovery of available formats
    - Lookup of supported target formats for any source format
    - GUI dropdown population based on available conversions
    """
    
    _formats: Dict[FormatType, Type["Format"]] = {}
    _conversions: Dict[Tuple[FormatType, FormatType], Type["Conversion"]] = {}
    _display_names: Dict[FormatType, str] = {
        FormatType.BINMASK: "Binary Mask",
        FormatType.YOLO: "YOLO",
        FormatType.LABEL_STUDIO: "Label Studio",
        FormatType.ULTRALYTICS: "Ultralytics",
    }
    
    @classmethod
    def register_format(cls, format_type: FormatType, format_class: Type["Format"]) -> None:
        """Register a format class for a given format type."""
        cls._formats[format_type] = format_class
    
    @classmethod
    def register_conversion(
        cls, 
        source: FormatType, 
        target: FormatType, 
        conversion_class: Type["Conversion"]
    ) -> None:
        """Register a conversion between two format types."""
        cls._conversions[(source, target)] = conversion_class
    
    @classmethod
    def get_format_class(cls, format_type: FormatType) -> Optional[Type["Format"]]:
        """Get the format class for a given format type."""
        return cls._formats.get(format_type)
    
    @classmethod
    def get_conversion_class(
        cls, 
        source: FormatType, 
        target: FormatType
    ) -> Optional[Type["Conversion"]]:
        """Get the conversion class for a source-target pair."""
        return cls._conversions.get((source, target))
    
    @classmethod
    def get_supported_targets(cls, source: FormatType) -> List[FormatType]:
        """Get all formats that the source format can be converted to."""
        return [
            target for (src, target) in cls._conversions.keys() 
            if src == source
        ]
    
    @classmethod
    def get_all_source_formats(cls) -> List[FormatType]:
        """Get all formats that can be used as conversion sources."""
        return list(set(src for src, _ in cls._conversions.keys()))
    
    @classmethod
    def get_display_name(cls, format_type: FormatType) -> str:
        """Get human-readable display name for a format type."""
        return cls._display_names.get(format_type, format_type.name)
    
    @classmethod
    def get_format_choices(cls) -> List[Tuple[str, FormatType]]:
        """Get list of (display_name, format_type) tuples for GUI dropdowns."""
        sources = cls.get_all_source_formats()
        return [(cls.get_display_name(fmt), fmt) for fmt in sources]
    
    @classmethod
    def get_target_choices(cls, source: FormatType) -> List[Tuple[str, FormatType]]:
        """Get list of valid target formats for a given source format."""
        targets = cls.get_supported_targets(source)
        return [(cls.get_display_name(fmt), fmt) for fmt in targets]


def register_format(format_type: FormatType):
    """Decorator to register a format class with the registry."""
    def decorator(cls: Type["Format"]) -> Type["Format"]:
        FormatRegistry.register_format(format_type, cls)
        cls.format_type = format_type
        return cls
    return decorator


class Format(abc.ABC):
    """
    Abstract base class for dataset format validation.
    
    Subclasses must implement validate_structure() to check if a directory
    conforms to the expected format structure.
    
    Attributes:
        path: Path to the dataset directory
        format_type: The FormatType enum value (set by @register_format decorator)
    """
    
    format_type: FormatType = None  # Set by decorator
    
    def __init__(self, path: str | Path):
        """
        Initialize the format validator.
        
        Args:
            path: Path to the dataset directory to validate
        """
        self.path = Path(path)
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the dataset directory.
        
        Performs basic existence check, then delegates to format-specific validation.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.path.exists():
            return False, f"The path '{self.path}' does not exist."
        if not self.path.is_dir():
            return False, f"The path '{self.path}' is not a directory."
        return self.validate_structure()
    
    @abc.abstractmethod
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate the format-specific directory structure.
        
        Must be implemented by subclasses to check for required files and folders.
        
        Returns:
            Tuple of (success: bool, message: str describing the issue if failed)
        """
        pass
    
    @classmethod
    def get_display_name(cls) -> str:
        """Get the human-readable name for this format."""
        if cls.format_type:
            return FormatRegistry.get_display_name(cls.format_type)
        return cls.__name__
