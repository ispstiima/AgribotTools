"""
Ultralytics format definition and validation.

Ultralytics datasets expect the following structure:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/              # Optional
    │   ├── images/
    │   └── labels/
    └── <dataset_name>.yaml
"""

from pathlib import Path
from typing import Tuple, Optional, List
import yaml

from cvtoolkit.formats.format import Format, FormatType, register_format

@register_format(FormatType.ULTRALYTICS)
class Ultralytics(Format):
    """
    Base class for Ultralytics YOLO format validation.
    
    Provides common validation logic for both segmentation and detection variants.
    """
    
    REQUIRED_SPLITS = ["train", "val"]
    OPTIONAL_SPLITS = ["test"]
    
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate that the directory contains required Ultralytics structure.
        
        Checks for:
        - YAML configuration file
        - Required split directories (train, val)
        - images/ and labels/ subdirectories in each split
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check for YAML file
        yaml_file = self.find_yaml_file()
        if yaml_file is None:
            return False, f"No YAML configuration file found in '{self.path}'."
        
        # Validate YAML contents
        yaml_valid, yaml_msg = self._validate_yaml(yaml_file)
        if not yaml_valid:
            return False, yaml_msg
        
        # Check required split directories
        for split in self.REQUIRED_SPLITS:
            split_valid, split_msg = self._validate_split_dir(split)
            if not split_valid:
                return False, split_msg
        
        # Optionally validate test split if it exists
        test_dir = self.path / "test"
        if test_dir.exists():
            test_valid, test_msg = self._validate_split_dir("test")
            if not test_valid:
                return False, test_msg
        
        return True, ""
    
    def _validate_split_dir(self, split_name: str) -> Tuple[bool, str]:
        """Validate a split directory (train/val/test) structure."""
        split_dir = self.path / split_name
        
        if not split_dir.exists():
            return False, f"Missing required '{split_name}' directory in '{self.path}'."
        
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        if not images_dir.exists():
            return False, f"Missing 'images' directory in '{split_dir}'."
        
        if not labels_dir.exists():
            return False, f"Missing 'labels' directory in '{split_dir}'."
        
        # Check for matching image/label pairs
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_stems = {
            f.stem for f in images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        }
        label_stems = {
            f.stem for f in labels_dir.iterdir() 
            if f.is_file() and f.suffix.lower() == ".txt"
        }
        
        if not image_stems:
            return False, f"No image files found in '{images_dir}'."
        
        missing_labels = image_stems - label_stems
        if missing_labels and len(missing_labels) > len(image_stems) * 0.1:
            # Allow up to 10% missing labels (could be images without annotations)
            return False, f"Many label files missing in '{labels_dir}'. Missing: {len(missing_labels)}/{len(image_stems)}"
        
        return True, ""
    
    def _validate_yaml(self, yaml_path: Path) -> Tuple[bool, str]:
        """Validate the YAML configuration file."""
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                return False, f"YAML file '{yaml_path}' is empty."
            
            # Check for required keys
            if 'names' not in yaml_data:
                return False, f"YAML file missing 'names' key (class names)."
            
            if 'train' not in yaml_data and 'val' not in yaml_data:
                return False, f"YAML file missing 'train' or 'val' path specifications."
            
            return True, ""
        except yaml.YAMLError as e:
            return False, f"Invalid YAML file '{yaml_path}': {e}"
        except Exception as e:
            return False, f"Error reading YAML file '{yaml_path}': {e}"
    
    def find_yaml_file(self) -> Optional[Path]:
        """Find the dataset YAML configuration file."""
        yaml_files = list(self.path.glob('*.yaml')) + list(self.path.glob('*.yml'))
        return yaml_files[0] if yaml_files else None
    
    def read_yaml_data(self) -> Optional[dict]:
        """Read and parse the YAML configuration file."""
        yaml_file = self.find_yaml_file()
        if yaml_file is None:
            return None
        
        try:
            with open(yaml_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    
    def get_class_names(self) -> Optional[List[str]]:
        """Extract class names from the YAML configuration."""
        yaml_data = self.read_yaml_data()
        if yaml_data is None or 'names' not in yaml_data:
            return None
        
        names_data = yaml_data['names']
        if isinstance(names_data, dict):
            return [name for _, name in sorted(names_data.items())]
        elif isinstance(names_data, list):
            return names_data
        return None
    
    def get_splits(self) -> List[str]:
        """Get list of available splits in the dataset."""
        splits = []
        for split in self.REQUIRED_SPLITS + self.OPTIONAL_SPLITS:
            if (self.path / split).exists():
                splits.append(split)
        return splits

