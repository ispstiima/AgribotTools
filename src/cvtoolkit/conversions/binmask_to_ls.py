"""
Binary Mask to Label Studio format conversion.

This is a composite conversion that goes through YOLO as an intermediate format:
BinMask → YOLO → Label Studio
"""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.conversions.binmask_to_yolo import BinmaskToYolo
from cvtoolkit.conversions.yolo_to_ls import YoloToLabelStudio

log = logging.getLogger("BinmaskToLs")


@register_conversion(FormatType.BINMASK, FormatType.LABEL_STUDIO)
class BinmaskToLabelStudio(Conversion):
    """
    Base class for Binary Mask to Label Studio conversions.
    
    This performs a two-step conversion:
    1. BinMask → YOLO (in a temporary directory)
    2. YOLO → Label Studio
    """
    
    def convert(self, **ls_kwargs) -> Path:
        """
        Convert binary mask dataset to Label Studio format.
        
        Args:
            **ls_kwargs: Additional arguments for the YOLO to LS conversion
                (to_name, from_name, out_type, image_root_url, image_ext, image_dims)
        
        Returns:
            Path to the Label Studio dataset
        """
        with TemporaryDirectory() as temp_dir:
            temp_yolo = Path(temp_dir) / "yolo_temp"
            
            log.info("Step 1: Converting BinMask → YOLO...")
            
            # Step 1: BinMask → YOLO
            yolo_converter = BinmaskToYolo(self.source_path, temp_yolo)
            yolo_converter.convert()
            
            log.info("Step 2: Converting YOLO → Label Studio...")
            
            # Step 2: YOLO → Label Studio  
            ls_converter = YoloToLabelStudio(temp_yolo, self.target_path)
            self._track_path(self.target_path)
            result = ls_converter.convert(**ls_kwargs)
            
            log.info(f"Conversion complete: {result}")
            
            return result

