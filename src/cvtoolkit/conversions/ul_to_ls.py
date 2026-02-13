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
from typing import Optional, Tuple
from cvtoolkit.formats.format import FormatType
from cvtoolkit.formats.ls import DEFAULT_IMAGE_ROOT_URL
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.conversions.ul_to_yolo import UltralyticsToYolo
from cvtoolkit.conversions.yolo_to_ls import YoloToLabelStudio


log = logging.getLogger("UlToLs")


@register_conversion(FormatType.ULTRALYTICS, FormatType.LABEL_STUDIO)
class UltralyticsToLabelStudio(Conversion):
    """Base class for Ultralytics to Label Studio conversions."""
    
    def convert(
        self,
        to_name: str = "image",
        from_name: str = "label",
        out_type: str = "annotations",
        image_root_url: str = DEFAULT_IMAGE_ROOT_URL,
        image_ext: str = ".jpg,.png",
        image_dims: Optional[Tuple[int, int]] = None,
    ) -> Path:
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
            self._report_progress(0.20, "Step 1: Converting Ultralytics → YOLO...")
            ul_to_yolo = UltralyticsToYolo(self.source_path, temp_yolo_path, self.task_type)
            ul_to_yolo.set_progress_callback(self._sub_progress_callback(0.20, 0.55))
            ul_to_yolo.convert()
            
            # Step 2: YOLO -> Label Studio
            self._report_progress(0.55, "Step 2: Converting YOLO → Label Studio...")
            self._track_path(self.target_path)
            yolo_to_ls = YoloToLabelStudio(temp_yolo_path, self.target_path, self.task_type)
            yolo_to_ls.set_progress_callback(self._sub_progress_callback(0.55, 0.95))
            yolo_to_ls.convert(
                to_name=to_name,
                from_name=from_name,
                out_type=out_type,
                image_root_url=image_root_url,
                image_ext=image_ext,
                image_dims=image_dims,
            )
        
        log.info(f"Conversion complete: {self.target_path}")
        return self.target_path

