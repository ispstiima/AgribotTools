"""
Mask utilities for converting between binary masks and YOLO format.
"""

import cv2
import logging
import numpy as np
from cvtoolkit.formats import TaskType
from cvtoolkit.formats.yolo import seg_to_bbox

log = logging.getLogger("MaskToYolo")


def mask_to_yolo(mask: np.ndarray, task_type: TaskType = TaskType.GENERIC):
    """Convert a binary mask to YOLO segmentation or bounding box format.

    Args:
        mask: Binary mask where 255 represents the object and 0 represents the background.
        task_type: The type of annotation to generate (SEGMENTATION or DETECTION).

    Returns:
        A list containing YOLO-formatted segmentation or bounding box annotations.
        Each annotation starts with class_id (0 for binary masks) followed by coordinates.
    """
    img_height, img_width = mask.shape

    yolo_lines = []

    # Find contours in the binary mask
    binary_mask = (mask == 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    log.info(f"Found {len(contours)} contours, task_type={task_type}")

    for idx, contour in enumerate(contours):
        log.info(f"  Contour {idx}: shape={contour.shape}, len={len(contour)}")

        if len(contour) < 3:
            log.info(f"    Skipping: contour has < 3 points")
            continue
        
        contour = contour.squeeze()
        log.info(f"    After squeeze: shape={contour.shape}")
        
        # Handle case where squeeze reduces to 1D (single point - shouldn't happen with >= 3)
        if contour.ndim == 1:
            log.warning(f"    Skipping: contour became 1D after squeeze")
            continue

        # Start with class ID 0 (binary masks have single class)
        seg_line = [0]

        for point in contour:
            seg_line.append(round(point[0] / img_width, 6))
            seg_line.append(round(point[1] / img_height, 6))

        if task_type == TaskType.SEGMENTATION:
            yolo_lines.append(seg_line)
            log.info(f"    Added segmentation line with {len(seg_line)} elements")
        elif task_type == TaskType.DETECTION:
            bbox_line = seg_to_bbox(seg_line)
            yolo_lines.append(bbox_line)
            log.info(f"    Added detection bbox: {bbox_line}")
        else:
            log.info(f"    Skipping: task_type={task_type} not SEGMENTATION or DETECTION")
            continue

    return yolo_lines
