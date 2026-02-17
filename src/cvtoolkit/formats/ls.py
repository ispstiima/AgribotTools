"""
Label Studio format definition and validation.

Label Studio datasets expect the following structure:
    dataset/
    ├── images/           # Directory containing image files
    ├── task.json         # JSON file with annotation tasks
    └── template.label_config.xml  # Labeling configuration (optional)
"""

from pathlib import Path
from typing import Tuple
from cvtoolkit.formats.format import Format, FormatType, register_format


DEFAULT_IMAGE_ROOT_URL = "/data/local-files/?d=images"


@register_format(FormatType.LABEL_STUDIO)
class LabelStudio(Format):
    """
    Label Studio format definition and validation.
    
    Label Studio datasets expect the following structure:
        dataset/
        ├── images/           # Directory containing image files
        ├── task.json         # JSON file with annotation tasks
        └── template.label_config.xml  # Labeling configuration (optional)
    """
    
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate that the directory contains required Label Studio structure.
        
        Checks for:
        - images/ directory with at least one image file
        - At least one .json task file
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        images_dir = self.path / "images"
        
        # Check images directory
        if not images_dir.exists():
            return False, f"Missing 'images' directory in '{self.path}'."
        
        if not images_dir.is_dir():
            return False, f"'images' is not a directory in '{self.path}'."
        
        # Check for at least one image
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            return False, f"No image files found in '{images_dir}'."
        
        # Check for JSON task file
        json_files = list(self.path.glob("*.json"))
        if not json_files:
            return False, f"No JSON task file found in '{self.path}'."
        
        return True, ""
    
    def get_task_file(self) -> Path | None:
        """Get the first JSON task file in the dataset directory."""
        json_files = list(self.path.glob("*.json"))
        return json_files[0] if json_files else None
    
    def get_config_file(self) -> Path | None:
        """Get the label configuration XML file if it exists."""
        xml_files = list(self.path.glob("*.xml"))
        return xml_files[0] if xml_files else None

