"""
Label Studio to Ultralytics format conversion.

Converts Label Studio annotation format (images/, task.json) to
Ultralytics YOLO format (train/val/test splits with .yaml config).

This is implemented as a two-step conversion:
1. Label Studio -> YOLO (intermediate)
2. YOLO -> Ultralytics
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.conversions.ls_to_yolo import LabelStudioToYolo
from cvtoolkit.conversions.yolo_to_ul import YoloToUltralytics


log = logging.getLogger("LsToUl")


class LabelStudioToUltralytics(Conversion):
    """Base class for Label Studio to Ultralytics conversions."""
    
    def convert(
        self,
        split_ratios: Tuple[float, float, Optional[float]] = (0.8, 0.2),
        include_test_split: bool = False,
        image_ext: str = ".jpg,.png",
        yaml_filename: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Path:
        """
        Convert Label Studio dataset to Ultralytics format.
        
        This performs a two-step conversion via an intermediate YOLO format.
        
        Args:
            split_ratios: Train/val/(test) split ratios
            include_test_split: Whether to create a test split
            image_ext: Comma-separated list of image extensions
            yaml_filename: Name for the output YAML file
            random_seed: Seed for reproducible splits
        
        Returns:
            Path to the Ultralytics dataset
        """
        log.info(f"Converting Label Studio to Ultralytics: {self.source_path} -> {self.target_path}")
        
        # Create a temporary directory for the intermediate YOLO format
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_yolo_path = Path(temp_dir) / "yolo_intermediate"
            
            # Step 1: Label Studio -> YOLO
            log.info("Step 1: Converting Label Studio to YOLO (intermediate)...")
            ls_to_yolo = LabelStudioToYolo(self.source_path, temp_yolo_path, self.task_type)
            ls_to_yolo.convert(image_ext=image_ext)
            
            # Step 2: YOLO -> Ultralytics
            log.info("Step 2: Converting YOLO to Ultralytics...")
            self._track_path(self.target_path)
            yolo_to_ul = YoloToUltralytics(self.source_path, temp_yolo_path, self.task_type)
            yolo_to_ul.convert(
                split_ratios=split_ratios,
                include_test_split=include_test_split,
                image_ext=image_ext,
                yaml_filename=yaml_filename,
                random_seed=random_seed,
            )
        
        log.info(f"Conversion complete: {self.target_path}")
        return self.target_path

