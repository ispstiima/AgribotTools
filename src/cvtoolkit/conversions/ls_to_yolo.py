"""
Label Studio to YOLO format conversion.

Converts Label Studio annotation format (images/, task.json) to
YOLO format (images/, labels/, classes.txt).
"""

import json
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from tqdm import tqdm
from cvtoolkit.formats import TaskType
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.formats.yolo import save_yolo_file
from file_utils import copy_files_monitored


log = logging.getLogger("LsToYolo")


def parse_bbox_value(value: dict, img_w: int, img_h: int) -> list | None:
    """
    Convert Label Studio bounding box annotation to YOLO format.
    
    Args:
        value: Dictionary containing Label Studio annotation information
        img_w: Image width
        img_h: Image height
    
    Returns:
        YOLO format list [class_id, x_center, y_center, width, height] or None
    """
    if not ("x" in value and "y" in value and "width" in value and "height" in value):
        return None

    class_id = 0  # TODO: implement multi-class support

    w = value["width"]
    h = value["height"]

    x = (value["x"] + w / 2) / 100
    y = (value["y"] + h / 2) / 100
    w = w / 100
    h = h / 100

    return [class_id, x, y, w, h]


def parse_seg_value(value: dict, img_w: int, img_h: int) -> list:
    """
    Convert Label Studio segmentation mask annotation to YOLO format.
    
    Args:
        value: Dictionary containing annotation information including RLE-encoded mask
        img_w: Image width
        img_h: Image height
    
    Returns:
        YOLO format list [class_id, x1, y1, x2, y2, ...]
    """
    from cvtoolkit.rle import decode_rle
    from cvtoolkit.mask import mask_to_yolo
    
    class_id = 0  # TODO: implement multi-class support

    flat_binmask = decode_rle(value["rle"])
    binmask = np.reshape(flat_binmask, [img_h, img_w, 4])[:, :, 3]
    seg_lines = mask_to_yolo(binmask, TaskType.SEGMENTATION)

    seg_line = [class_id]

    # Merge multiple found contours into one if needed
    if len(seg_lines) > 1:
        seg_line.extend(list(chain(*[line[1:] for line in seg_lines])))
    else:
        seg_line.extend(seg_lines[0][1:] if seg_lines else [])

    return seg_line


@register_conversion(FormatType.LABEL_STUDIO, FormatType.YOLO)
class LabelStudioToYolo(Conversion):
    """Base class for Label Studio to YOLO conversions."""
    
    def convert(self, image_ext: str = ".jpg,.png") -> Path:
        """
        Perform the Label Studio to YOLO conversion.
        
        Args:
            image_ext: Comma-separated list of valid image extensions
        
        Returns:
            Path to the YOLO dataset
        """
        self._report_progress(0.25, "Reading JSON & creating directories...")
        
        # Create target directories
        labels_path = self.target_path / "labels"
        labels_path.mkdir(parents=True, exist_ok=True)
        self._track_path(self.target_path)
        
        # Get source paths
        ls_images = self.source_path / "images"
        json_files = list(self.source_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON task file found in {self.source_path}")
        
        json_path = json_files[0]
        
        log.info(f"Reading LS JSON from: {json_path}")
        
        with json_path.open("r", encoding="utf-8") as f:
            tasks = json.load(f)
        
        if tasks is None:
            raise ValueError("No tasks found in the task file.")
        
        image_ext_list = [x.strip().lower() for x in image_ext.split(",")]
        parse_value = parse_bbox_value if self.task_type == TaskType.DETECTION else parse_seg_value
        total_tasks = len(tasks)
        
        for i, task in enumerate(tqdm(tasks, ascii="░▒█", desc="Converting LS to YOLO")):
            image_filename = task["data"]["image"].split("/")[-1]
            image_path = ls_images / image_filename
            
            if not image_path.exists():
                log.warning(f"Missing image file: {image_path}")
                continue
            
            if image_path.suffix.lower() not in image_ext_list:
                log.warning(f"Unsupported image format: {image_path.suffix}")
                continue
            
            label_filename = f"{image_path.stem}.txt"
            label_path = labels_path / label_filename
            
            yolo_lines = []
            
            for annotation in task.get("annotations", []):
                for result in annotation.get("result", []):
                    img_w, img_h = result["original_width"], result["original_height"]
                    value = result["value"]
                    
                    yolo_line = parse_value(value, img_w, img_h)
                    if yolo_line:
                        yolo_lines.append(' '.join([str(x) for x in yolo_line]))
            
            with label_path.open("w", encoding="utf-8") as f:
                for line in yolo_lines:
                    f.write(line + "\n")
            
            # Report progress (0.30 to 0.80 range for task conversion)
            progress = 0.30 + (0.50 * (i + 1) / total_tasks)
            self._report_progress(progress, f"Converting task {i + 1}/{total_tasks}")
        
        log.info(f"Saved YOLO annotations to: {labels_path}")
        
        self._report_progress(0.85, "Writing classes.txt...")
        
        # Create classes.txt (default single class for now)
        classes_path = self.target_path / "classes.txt"
        with classes_path.open("w", encoding="utf-8") as f:
            f.write("object\n")  # TODO: extract class names from LS config
        
        self._report_progress(0.90, "Copying images...")
        
        # Copy images
        yolo_images = self.target_path / "images"
        copy_files_monitored(ls_images, yolo_images, desc="Copying images from LS")
        
        return self.target_path

