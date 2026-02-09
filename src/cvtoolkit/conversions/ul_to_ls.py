"""
Ultralytics to Label Studio format conversion.

Converts Ultralytics YOLO format (train/val/test splits with .yaml) to
Label Studio annotation format (images/, task.json).

This is implemented as a two-step conversion:
1. Ultralytics -> YOLO (intermediate)
2. YOLO -> Label Studio
"""

import logging
import tempfile
from pathlib import Path
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.conversions.ul_to_yolo import UltralyticsToYolo
from cvtoolkit.conversions.yolo_to_ls import YoloToLabelStudio


log = logging.getLogger("UlToLs")


@register_conversion(FormatType.ULTRALYTICS, FormatType.LABEL_STUDIO)
class UltralyticsToLabelStudio(Conversion):
    """Base class for Ultralytics to Label Studio conversions."""
    
    def convert(self, image_ext: str = ".jpg,.png") -> Path:
        """
        Convert Ultralytics dataset to Label Studio format.
        
        This performs a two-step conversion via an intermediate YOLO format.
        
        Args:
            image_ext: Comma-separated list of valid image extensions
        
        Returns:
            Path to the Label Studio dataset
        """
        log.info(f"Converting Ultralytics to Label Studio: {self.source_path} -> {self.target_path}")
        
        # Create a temporary directory for the intermediate YOLO format
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_yolo_path = Path(temp_dir) / "yolo_intermediate"
            
            # Step 1: Ultralytics -> YOLO
            log.info("Step 1: Converting Ultralytics to YOLO (intermediate)...")
            ul_to_yolo = UltralyticsToYolo(self.source_path, temp_yolo_path)
            ul_to_yolo.convert()
            
            # Step 2: YOLO -> Label Studio
            log.info("Step 2: Converting YOLO to Label Studio...")
            self._track_path(self.target_path)
            yolo_to_ls = YoloToLabelStudio(temp_yolo_path, self.target_path, self.task_type)
            yolo_to_ls.convert(image_ext=image_ext)
        
        log.info(f"Conversion complete: {self.target_path}")
        return self.target_path

