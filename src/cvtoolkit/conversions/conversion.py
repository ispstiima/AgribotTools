"""
Conversion base classes and registry for format conversions.

This module provides:
- Abstract base classes for conversions
- Decorator for registering conversions with the format registry
- Rollback support for failed conversions
"""

import abc
import shutil
import logging
from pathlib import Path
from typing import Type, Optional, Tuple, Any, Callable
from contextlib import contextmanager
from cvtoolkit.formats.format import FormatType, FormatRegistry
from cvtoolkit.formats import TaskType

log = logging.getLogger("Conversion")


def register_conversion(source: FormatType, target: FormatType):
    """
    Decorator to register a conversion class with the format registry.
    
    Args:
        source: The source format type
        target: The target format type
    
    Usage:
        @register_conversion(FormatType.YOLO_SEG, FormatType.LABEL_STUDIO_SEG)
        class YoloSegToLabelStudioSeg(Conversion):
            ...
    """
    def decorator(cls: Type["Conversion"]) -> Type["Conversion"]:
        FormatRegistry.register_conversion(source, target, cls)
        cls.source_type = source
        cls.target_type = target
        return cls
    return decorator


class ConversionError(Exception):
    """Exception raised when a conversion fails."""
    pass


class Conversion(abc.ABC):
    """
    Abstract base class for all format conversions.
    
    Provides:
    - Source/target path management
    - Validation before conversion
    - Rollback support on failure
    - Progress tracking hooks
    
    Attributes:
        source_path: Path to the source dataset
        target_path: Path where converted dataset will be saved
        source_type: FormatType of the source (set by decorator)
        target_type: FormatType of the target (set by decorator)
    """
    
    source_type: FormatType = None  # Set by decorator
    target_type: FormatType = None  # Set by decorator
    
    def __init__(self, source_path: str | Path, target_path: str | Path, task_type: TaskType=TaskType.GENERIC):
        """
        Initialize the conversion.
        
        Args:
            source_path: Path to the source dataset directory
            target_path: Path where the converted dataset will be created
            task_type: Type of the Object Detection task described by the target dataset
        """
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.task_type = task_type
        self._rollback_paths: list[Path] = []
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """
        Set a callback for progress updates.
        
        Args:
            callback: Function that receives (progress: float 0-1, message: str)
        """
        self._progress_callback = callback
    
    def _report_progress(self, progress: float, message: str) -> None:
        """Report progress to the callback if set."""
        if self._progress_callback:
            self._progress_callback(progress, message)
    
    def validate_source(self) -> Tuple[bool, str]:
        """
        Validate the source dataset.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        format_class = FormatRegistry.get_format_class(self.source_type)
        if format_class is None:
            return False, f"No format validator registered for {self.source_type}"
        
        validator = format_class(self.source_path)
        return validator.validate()
    
    def validate_target_path(self) -> Tuple[bool, str]:
        """
        Validate that the target path is suitable for output.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.target_path.exists():
            if not self.target_path.is_dir():
                return False, f"Target path '{self.target_path}' exists but is not a directory."
            if any(self.target_path.iterdir()):
                return False, f"Target path '{self.target_path}' already exists and is not empty."
        return True, ""
    
    @contextmanager
    def _rollback_context(self):
        """
        Context manager that tracks created paths and rolls back on exception.
        
        Usage:
            with self._rollback_context():
                self._track_path(new_dir)
                # ... do conversion work ...
        """
        self._rollback_paths = []
        try:
            yield
        except Exception as e:
            log.error(f"Conversion failed: {e}. Rolling back...")
            self._rollback()
            raise ConversionError(f"Conversion failed: {e}") from e
    
    def _track_path(self, path: Path) -> None:
        """Track a path for potential rollback."""
        self._rollback_paths.append(path)
    
    def _rollback(self) -> None:
        """Remove all tracked paths in reverse order."""
        for path in reversed(self._rollback_paths):
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        log.info(f"Rolled back directory: {path}")
                    else:
                        path.unlink()
                        log.info(f"Rolled back file: {path}")
            except Exception as e:
                log.warning(f"Failed to rollback {path}: {e}")
        self._rollback_paths = []
    
    def run(self, **kwargs) -> Path:
        """
        Execute the conversion with validation and rollback support.
        
        This is the main entry point that:
        1. Validates the source dataset
        2. Validates the target path
        3. Runs the conversion
        4. Rolls back on failure
        
        Args:
            **kwargs: Additional arguments passed to convert()
        
        Returns:
            Path to the converted dataset
        
        Raises:
            ConversionError: If validation or conversion fails
        """
        # Validate source
        self._report_progress(0.0, "Validating source dataset...")
        valid, message = self.validate_source()
        if not valid:
            raise ConversionError(f"Source validation failed: {message}")
        
        # Validate target
        self._report_progress(0.1, "Validating target path...")
        valid, message = self.validate_target_path()
        if not valid:
            raise ConversionError(f"Target validation failed: {message}")
        
        # Run conversion with rollback support
        self._report_progress(0.2, "Starting conversion...")
        with self._rollback_context():
            result = self.convert(**kwargs)
        
        self._report_progress(1.0, "Conversion complete!")
        return result
    
    @abc.abstractmethod
    def convert(self, **kwargs) -> Path:
        """
        Perform the actual conversion.
        
        Must be implemented by subclasses. Should use self._track_path()
        to register created directories/files for rollback.
        
        Args:
            **kwargs: Conversion-specific arguments
        
        Returns:
            Path to the converted dataset
        """
        pass
    
    @classmethod
    def get_display_name(cls) -> str:
        """Get a human-readable name for this conversion."""
        if cls.source_type and cls.target_type:
            src = FormatRegistry.get_display_name(cls.source_type)
            tgt = FormatRegistry.get_display_name(cls.target_type)
            return f"{src} → {tgt}"
        return cls.__name__


class ReversibleConversion(Conversion):
    """
    Base class for conversions that can be reversed.
    
    Subclasses should implement both convert() and reverse_convert().
    """
    
    @abc.abstractmethod
    def reverse_convert(self, **kwargs) -> Path:
        """
        Perform the reverse conversion.
        
        Args:
            **kwargs: Conversion-specific arguments
        
        Returns:
            Path to the converted dataset
        """
        pass
    
    def run_reverse(self, **kwargs) -> Path:
        """
        Execute the reverse conversion with validation and rollback.
        
        Note: This swaps source and target for validation purposes.
        
        Args:
            **kwargs: Additional arguments passed to reverse_convert()
        
        Returns:
            Path to the converted dataset
        """
        # Swap source and target types for validation
        original_source = self.source_type
        original_target = self.target_type
        self.source_type = original_target
        self.target_type = original_source
        
        # Swap paths
        self.source_path, self.target_path = self.target_path, self.source_path
        
        try:
            return self.run(**kwargs)
        finally:
            # Restore original types and paths
            self.source_type = original_source
            self.target_type = original_target
            self.source_path, self.target_path = self.target_path, self.source_path
    
    def convert(self, **kwargs) -> Path:
        """Default implementation calls the forward conversion."""
        return self._do_forward_convert(**kwargs)
    
    @abc.abstractmethod
    def _do_forward_convert(self, **kwargs) -> Path:
        """Implement the forward conversion logic here."""
        pass

