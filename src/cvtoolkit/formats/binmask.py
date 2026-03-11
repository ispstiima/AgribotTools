"""
Binary Mask format definition and validation.

Binary mask datasets expect the following structure:
    dataset/
    ├── images/        # Directory containing original image files
    ├── labels/        # Directory containing binary mask images (same names as images)
    └── classes.txt    # File listing class names (one per line)

Binary masks should be binary images where:
- 255 (white) represents the object/foreground
- 0 (black) represents the background
"""

from pathlib import Path
from typing import Tuple, List, Optional
from cvtoolkit.formats.format import Format, FormatType, register_format


@register_format(FormatType.BINMASK)
class Binmask(Format):
    """
    Binary mask format for segmentation datasets.
    
    Binary masks are binary images where each pixel indicates whether
    it belongs to the foreground (255) or background (0).
    """
    
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate that the directory contains required binary mask structure.
        
        Checks for:
        - images/ directory with image files
        - labels/ directory with mask files
        - classes.txt file (optional but recommended)
        - Matching image/mask file pairs
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        images_dir = self.path / "images"
        labels_dir = self.path / "labels"
        
        # Check required directories
        if not images_dir.exists():
            return False, f"Missing 'images' directory in '{self.path}'."
        
        if not labels_dir.exists():
            return False, f"Missing 'labels' directory in '{self.path}'."
        
        # Get file stems
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        mask_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        
        image_stems = {
            f.stem for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        }
        mask_stems = {
            f.stem for f in labels_dir.iterdir()
            if f.is_file() and f.suffix.lower() in mask_extensions
        }
        
        if not image_stems:
            return False, f"No image files found in '{images_dir}'."
        
        if not mask_stems:
            return False, f"No mask files found in '{labels_dir}'."
        
        # Check for matching files
        missing_masks = image_stems - mask_stems
        missing_images = mask_stems - image_stems
        
        if missing_masks:
            return False, f"Missing binary masks in '{labels_dir}' for {len(missing_masks)} images."
        
        if missing_images:
            return False, f"Missing images in '{images_dir}' for {len(missing_images)} masks."
        
        return True, ""
    
    def get_classes(self) -> Optional[dict]:
        """Get class names from classes.txt if it exists."""
        classes_file = self.path / "classes.txt"
        if not classes_file.exists():
            return None
        
        try:
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            return {i: name for i, name in enumerate(class_names)}
        except Exception:
            return None
    
    def get_image_paths(self) -> List[Path]:
        """Get list of original image file paths."""
        images_dir = self.path / "images"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return [
            f for f in images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    
    def get_mask_path(self, image_path: Path) -> Optional[Path]:
        """Get the mask file path for a given image."""
        labels_dir = self.path / "labels"
        mask_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
        
        for ext in mask_extensions:
            mask_path = labels_dir / f"{image_path.stem}{ext}"
            if mask_path.exists():
                return mask_path
        return None
    
    def get_mask_paths(self) -> List[Path]:
        """Get list of mask file paths."""
        labels_dir = self.path / "labels"
        mask_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return [
            f for f in labels_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in mask_extensions
        ]
