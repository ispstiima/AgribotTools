"""
YOLO to Label Studio format conversion.

Converts YOLO annotation format (images/, labels/, classes.txt) to
Label Studio format (images/, task.json, template.label_config.xml).
"""

import json
import uuid
import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import pathname2url
from PIL import Image
from tqdm import tqdm
from cvtoolkit.formats import TaskType
from cvtoolkit.formats.format import FormatType
from cvtoolkit.formats.ls import DEFAULT_IMAGE_ROOT_URL
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.colors import COLORS
from file_utils import copy_files_monitored


log = logging.getLogger("YoloToLs")


# Templates for Label Studio configuration
LABELS_TEMPLATE = """
  <{tag_name} name="{from_name}" toName="image">
{labels}  </{tag_name}>
"""

LABELING_CONFIG_TEMPLATE = """<View>
  <Image name="{to_name}" value="$image"/>
{body}</View>
"""


def generate_label_config(
    config_path: Path, 
    categories: dict, 
    tags: dict, 
    to_name: str = "image"
) -> str:
    """
    Generate a Label Studio labeling configuration XML file.
    
    Args:
        config_path: Path where the configuration file will be saved
        categories: Dictionary mapping category IDs to names
        tags: Dictionary mapping tag names to their types
        to_name: Name of the object to be labeled
    
    Returns:
        The generated XML configuration string
    """
    labels = ""
    for key in sorted(categories.keys()):
        color = COLORS[int(key) % len(COLORS)]
        label = f'    <Label value="{categories[key]}" background="rgba({color[0]}, {color[1]}, {color[2]}, 1)"/>\n'
        labels += label

    body = ""
    for from_name in tags:
        tag_body = LABELS_TEMPLATE.format(
            tag_name=tags[from_name],
            labels=labels,
            from_name=from_name
        )
        body += f'\n  <Header value="{tags[from_name]}"/>' + tag_body

    config = LABELING_CONFIG_TEMPLATE.format(body=body, to_name=to_name)
    config_path.write_text(config)
    log.info(f"Label configuration file saved to: {config_path}")

    return config


def build_bbox_value(
    line: str, 
    categories: dict, 
    image_width: int, 
    image_height: int
) -> dict:
    """
    Build a Label Studio annotation from a YOLO bounding box line.
    
    Args:
        line: YOLO format annotation line (class_id x y w h [score])
        categories: Dictionary mapping category IDs to names
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Label Studio compatible annotation dictionary
    """
    values = line.split()
    label_id, x, y, width, height = values[0:5]
    score = float(values[5]) if len(values) >= 6 else None

    x, y, width, height = float(x), float(y), float(width), float(height)

    item = {
        "id": uuid.uuid4().hex[0:10],
        "type": "rectanglelabels",
        "value": {
            "x": (x - width / 2) * 100,
            "y": (y - height / 2) * 100,
            "width": width * 100,
            "height": height * 100,
            "rotation": 0,
            "rectanglelabels": [categories[int(label_id)]],
        },
    }

    if score:
        item["score"] = score

    return item


def build_seg_value(
    line: str, 
    categories: dict, 
    image_width: int, 
    image_height: int
) -> dict:
    """
    Build a Label Studio annotation from a YOLO segmentation line.
    
    Converts a YOLO polygon mask to a brush RLE annotation for Label Studio.
    
    Args:
        line: YOLO format annotation line (class_id x1 y1 x2 y2 ...)
        categories: Dictionary mapping category IDs to names
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        Label Studio compatible annotation dictionary
    """
    from cvtoolkit.rle import yolo_to_mask, mask_to_rle
    
    values = line.split()
    label_id = int(values[0])
    mask_class = categories[label_id]
    mask = yolo_to_mask(values[1:], image_width, image_height)
    rle = mask_to_rle(mask)

    item = {
        "id": str(uuid.uuid4())[0:8],
        "type": "brushlabels",
        "value": {"rle": rle, "format": "rle", "brushlabels": [mask_class]},
        "origin": "manual",
    }

    return item


@register_conversion(FormatType.YOLO, FormatType.LABEL_STUDIO)
class YoloToLabelStudio(Conversion):
    """Base class for YOLO to Label Studio conversions."""
    
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
        Perform the YOLO to Label Studio conversion.
        
        Args:
            to_name: Object name in Label Studio config
            from_name: Control tag name in Label Studio config
            out_type: Annotation type ("annotations" or "predictions")
            image_root_url: Root URL path where images will be hosted
            image_ext: Comma-separated list of image extensions
            image_dims: Optional tuple (width, height) if all images have same dimensions
        
        Returns:
            Path to the Label Studio dataset
        """
        self._report_progress(0.25, "Creating output directories & loading classes...")
        
        # Create target directory
        self.target_path.mkdir(parents=True, exist_ok=True)
        self._track_path(self.target_path)
        
        # Load classes
        classes_path = self.source_path / "classes.txt"
        with classes_path.open() as f:
            lines = [line.strip() for line in f.readlines()]
        categories = {i: line for i, line in enumerate(lines)}
        
        # Setup paths
        json_path = self.target_path / "task.json"
        config_path = self.target_path / "template.label_config.xml"
        label_tag = (
            "RectangleLabels" if self.task_type == TaskType.DETECTION else
            "BrushLabels" if self.task_type == TaskType.SEGMENTATION else
            None
        )
        
        self._report_progress(0.30, "Generating label config...")
        
        # Generate label config
        generate_label_config(config_path, categories, {from_name: label_tag}, to_name)
        
        yolo_labels = self.source_path / "labels"
        yolo_images = self.source_path / "images"
        
        image_ext_list = [x.strip().lower() for x in image_ext.split(",")]
        build_value = (
            build_bbox_value if self.task_type == TaskType.DETECTION else
            build_seg_value if self.task_type == TaskType.SEGMENTATION else
            None
        )
        
        # Get image list
        image_list = [
            img for img in yolo_images.iterdir() 
            if img.suffix.lower() in image_ext_list
        ]
        total_images = len(image_list)
        
        tasks = []
        for i, image_path in enumerate(tqdm(image_list, ascii="░▒█", desc="Converting YOLO to LS")):
            url = image_root_url + ("" if image_root_url.endswith("/") else "/")
            task = {
                "data": {
                    "image": url + pathname2url(image_path.name)
                }
            }
            
            label_path = yolo_labels / f"{image_path.stem}.txt"
            
            if label_path.exists():
                task[out_type] = [{"result": [], "ground_truth": False}]
                
                if image_dims is None:
                    with Image.open(image_path) as im:
                        image_width, image_height = im.size
                else:
                    image_width, image_height = image_dims
                
                info = {
                    "to_name": to_name,
                    "from_name": from_name,
                    "image_rotation": 0,
                    "original_width": image_width,
                    "original_height": image_height,
                }
                
                with label_path.open("r") as f:
                    lines = f.readlines()
                
                if lines:
                    for line in lines:
                        item = build_value(line, categories, image_width, image_height)
                        item = {**item, **info}
                        task[out_type][0]["result"].append(item)
            
            tasks.append(task)
            
            # Report progress (0.35 to 0.85 range for image conversion)
            progress = 0.35 + (0.50 * (i + 1) / total_images)
            self._report_progress(progress, f"Converting image {i + 1}/{total_images}")
        
        self._report_progress(0.90, "Saving JSON...")
        
        # Save JSON
        if tasks:
            with json_path.open("w") as out:
                json.dump(tasks, out, indent=4)
            log.info(f"Saved {len(tasks)} tasks to {json_path}")
        
        self._report_progress(0.95, "Copying images...")
        
        # Copy images
        ls_images = self.target_path / "images"
        copy_files_monitored(yolo_images, ls_images, desc="Copying images")
        
        return self.target_path

