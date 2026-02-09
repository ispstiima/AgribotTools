"""
Ultralytics to YOLO format conversion.

Converts Ultralytics YOLO format (train/val/test splits with .yaml) back to
standard YOLO format (images/, labels/, classes.txt).
"""

import logging
import yaml
from pathlib import Path
from typing import Optional, List
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.formats.yolo import save_txt_file
from src.file_utils import copy_filtered_dir_monitored


log = logging.getLogger("UlToYolo")


def find_yaml_file(directory: Path) -> Optional[Path]:
    """Find a YAML configuration file in the directory."""
    yaml_files = list(directory.glob('*.yaml')) + list(directory.glob('*.yml'))
    if yaml_files:
        log.info(f"Found YAML file: {yaml_files[0]}")
        return yaml_files[0]
    return None


def read_yaml_data(yaml_path: Path) -> Optional[dict]:
    """Read and parse a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        if not yaml_data:
            log.error(f"YAML file is empty: {yaml_path}")
            return None
        return yaml_data
    except Exception as e:
        log.error(f"Error reading YAML file {yaml_path}: {e}")
        return None


def extract_class_names(yaml_data: dict) -> Optional[List[str]]:
    """Extract class names from YAML data."""
    if 'names' not in yaml_data:
        log.error("'names' key not found in YAML file")
        return None
    
    names_data = yaml_data['names']
    
    try:
        if isinstance(names_data, dict):
            return [name for _, name in sorted(names_data.items())]
        elif isinstance(names_data, list):
            return names_data
        else:
            log.error(f"Unexpected 'names' format in YAML: {type(names_data)}")
            return None
    except Exception as e:
        log.error(f"Error processing 'names' in YAML: {e}")
        return None


@register_conversion(FormatType.ULTRALYTICS, FormatType.YOLO)
class UltralyticsToYolo(Conversion):
    """Base class for Ultralytics to YOLO conversions."""
    
    def convert(self) -> Path:
        """
        Convert Ultralytics dataset back to standard YOLO format.
        
        Returns:
            Path to the YOLO dataset
        """
        log.info(f"Converting from Ultralytics ({self.source_path}) to YOLO ({self.target_path})")
        
        # Find and read YAML config
        yaml_path = find_yaml_file(self.source_path)
        if yaml_path is None:
            raise ValueError(f"No YAML configuration file found in {self.source_path}")
        
        yaml_data = read_yaml_data(yaml_path)
        if yaml_data is None:
            raise ValueError(f"Failed to read YAML file: {yaml_path}")
        
        class_names = extract_class_names(yaml_data)
        if class_names is None:
            raise ValueError("Failed to extract class names from YAML")
        
        # Create output directories
        yolo_images = self.target_path / "images"
        yolo_labels = self.target_path / "labels"
        
        self.target_path.mkdir(parents=True, exist_ok=True)
        yolo_images.mkdir(exist_ok=True)
        yolo_labels.mkdir(exist_ok=True)
        self._track_path(self.target_path)
        
        # Write classes.txt
        class_data = [[name] for name in class_names]
        save_txt_file("classes", self.target_path, class_data)
        log.info(f"Created classes.txt with {len(class_names)} classes")
        
        # Copy images and labels from all splits
        log.info("Copying files from split directories...")
        
        copy_filtered_dir_monitored(
            self.source_path, 
            yolo_images, 
            ".jpg,.png,.JPG,.PNG,.jpeg,.JPEG", 
            description="Copying images"
        )
        
        copy_filtered_dir_monitored(
            self.source_path, 
            yolo_labels, 
            ".txt", 
            description="Copying labels"
        )
        
        log.info(f"Conversion complete: {self.target_path}")
        
        return self.target_path

