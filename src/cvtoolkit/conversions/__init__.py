"""
CVToolkit conversions package.

Contains conversion classes for transforming between different annotation formats.
"""

from cvtoolkit.conversions.conversion import (
    Conversion,
    ReversibleConversion,
    ConversionError,
    register_conversion,
)

from cvtoolkit.conversions.binmask_to_yolo import BinmaskToYolo
from cvtoolkit.conversions.binmask_to_ls import BinmaskToLabelStudio
from cvtoolkit.conversions.yolo_to_ls import YoloToLabelStudio
from cvtoolkit.conversions.ls_to_yolo import LabelStudioToYolo
from cvtoolkit.conversions.yolo_to_ul import YoloToUltralytics
from cvtoolkit.conversions.ul_to_yolo import UltralyticsToYolo
from cvtoolkit.conversions.ls_to_ul import LabelStudioToUltralytics
from cvtoolkit.conversions.ul_to_ls import UltralyticsToLabelStudio

__all__ = [
    "Conversion",
    "ReversibleConversion",
    "ConversionError",
    "register_conversion",
    "BinmaskToYolo",
    "BinmaskToLabelStudio",
    "YoloToLabelStudio",
    "LabelStudioToYolo",
    "YoloToUltralytics",
    "UltralyticsToYolo",
    "LabelStudioToUltralytics",
    "UltralyticsToLabelStudio",
]
