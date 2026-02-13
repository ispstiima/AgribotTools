"""
CVToolkit formats package.

Contains format definitions and validators for different annotation formats.
"""

from enum import Enum
from cvtoolkit.formats.format import Format, FormatType, FormatRegistry, register_format
from cvtoolkit.formats.binmask import Binmask
from cvtoolkit.formats.ls import LabelStudio
from cvtoolkit.formats.ul import Ultralytics
from cvtoolkit.formats.yolo import Yolo


class TaskType(Enum):
    GENERIC = "Generic"
    DETECTION = "Detection"
    SEGMENTATION = "Segmentation"


__all__ = [
    "Format",
    "FormatType",
    "FormatRegistry",
    "TaskType",
    "register_format",
    "Binmask",
    "LabelStudio",
    "Ultralytics",
    "Yolo",
]
