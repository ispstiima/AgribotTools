"""
Binary Mask to YOLO format conversion.

Converts binary segmentation mask images to YOLO format for both
segmentation and bounding box annotation types.
"""

import cv2
import shutil
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.mask import mask_to_yolo
from cvtoolkit.formats.yolo import save_yolo_file
from src.file_utils import copy_files_monitored


log = logging.getLogger("BinmaskToYolo")


@register_conversion(FormatType.BINMASK, FormatType.YOLO)
class BinmaskToYolo(Conversion):
    """
    Base class for Binary Mask to YOLO conversions.
    
    Converts binary mask images where 255 = foreground and 0 = background
    to YOLO format annotations.
    """
    

    def convert(self) -> Path:
        """
        Convert binary mask dataset to YOLO format.
        
        Returns:
            Path to the YOLO dataset
        """
        images_dir = self.source_path / "images"
        masks_dir = self.source_path / "labels"
        classes_path = self.source_path / "classes.txt"
        
        self._report_progress(0.25, "Creating output directories...")
        
        # Create output directories
        yolo_images = self.target_path / "images"
        yolo_labels = self.target_path / "labels"
        
        yolo_images.mkdir(parents=True, exist_ok=True)
        yolo_labels.mkdir(parents=True, exist_ok=True)
        self._track_path(self.target_path)
        
        self._report_progress(0.3, "Copying images...")
        
        # Copy images
        copy_files_monitored(images_dir, yolo_images, dirs_exist_ok=True, desc="Copying images")
        
        # Copy classes.txt if exists
        if classes_path.exists():
            shutil.copy(classes_path, self.target_path / "classes.txt")
        else:
            # Create default classes.txt
            with open(self.target_path / "classes.txt", "w") as f:
                f.write("object\n")
        
        # Convert masks
        mask_paths = [p for p in masks_dir.iterdir() if p.is_file()]
        total_masks = len(mask_paths)
        
        self._report_progress(0.4, f"Converting {total_masks} masks...")
        
        for i, mask_file in enumerate(tqdm(mask_paths, ascii="░▒█", desc="Converting masks to YOLO")):
            if mask_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                log.warning(f"Unsupported mask format: {mask_file.suffix}")
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                log.warning(f"Could not read mask: {mask_file}")
                continue
            
            # Debug: log mask statistics
            unique_vals = np.unique(mask)
            log.info(f"Mask {mask_file.name}: shape={mask.shape}, unique values={unique_vals}")
            
            label_list = list(mask_to_yolo(mask, self.task_type))
            
            # Debug: log conversion result
            log.info(f"  -> Generated {len(label_list)} annotations for task_type={self.task_type}")

            if label_list is None:
                log.error(f"Could not convert mask: {mask_file}")
                raise ConversionError(f"Could not convert mask: {mask_file}")

            save_yolo_file(mask_file.stem, yolo_labels, label_list)
            
            # Report progress (0.4 to 0.95 range for mask conversion)
            progress = 0.4 + (0.55 * (i + 1) / total_masks)
            self._report_progress(progress, f"Converting mask {i + 1}/{total_masks}")
        
        log.info(f"Conversion complete: {self.target_path}")
        
        return self.target_path

