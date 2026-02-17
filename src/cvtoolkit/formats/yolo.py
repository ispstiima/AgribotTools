"""
YOLO format definition and validation.

Standard YOLO datasets expect the following structure:
    dataset/
    ├── images/        # Directory containing image files
    ├── labels/        # Directory containing .txt annotation files
    └── classes.txt    # File listing class names (one per line)
"""

from pathlib import Path
from typing import Tuple, List, Optional

from cvtoolkit.formats.format import Format, FormatType, register_format


def seg_to_bbox(seg_info: list) -> list:
    """Convert a segmentation label in YOLO format to a bounding box label.

    Args:
        seg_info: A list containing segmentation information.
                  - The first element is the class ID.
                  - The remaining elements are pairs of (x, y) coordinates defining the polygon.

    Returns:
        A bounding box representation in YOLO format as:
        [class_id, x_center, y_center, width, height]
        - class_id: The zero-based class index.
        - x_center: The normalized x-coordinate of the bbox center.
        - y_center: The normalized y-coordinate of the bbox center.
        - width: The normalized width of the bbox.
        - height: The normalized height of the bbox.
    """
    class_id, *points = seg_info

    points = [float(p) for p in points]

    x_min, y_min = min(points[0::2]), min(points[1::2])
    x_max, y_max = max(points[0::2]), max(points[1::2])

    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    bbox_info = [int(class_id), x_center, y_center, width, height]

    return bbox_info


def save_yolo_file(file_name: str, output_path: Path, data: list) -> None:
    """Saves YOLO-formatted data to text files.

    This function creates a text file in the specified output directory and writes each item
    in the data list as a space-separated line.

    Args:
        file_name: The base name for the output file (without extension).
        output_path: The directory where the file will be saved.
        data: A list of items to be written to the file. Each item will become a separate line.

    Raises:
        AssertionError: If data is empty, file_name is empty, or data is not a list.
    """
    assert file_name, "File name is empty"

    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{file_name}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        for item in data:
            line = " ".join(map(str, item))
            file.write(line + "\n")


def save_txt_file(file_name: str, output_path: Path, data: list) -> None:
    """Save data to a text file (generic version for classes.txt etc.)."""
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{file_name}.txt"
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            if isinstance(item, list):
                f.write(" ".join(map(str, item)) + "\n")
            else:
                f.write(str(item) + "\n")


def parse_classes(classes_path: Path) -> Optional[dict]:
    """Parse a YOLO classes file and return a dictionary mapping indices to names."""
    try:
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        if not class_names:
            return None
        return {i: name for i, name in enumerate(class_names)}
    except Exception:
        return None


@register_format(FormatType.YOLO)
class Yolo(Format):
    """
    Base class for YOLO format validation.
    
    Provides common validation logic for both segmentation and detection variants.
    """
    
    def validate_structure(self) -> Tuple[bool, str]:
        """
        Validate that the directory contains required YOLO structure.
        
        Checks for:
        - images/ directory
        - labels/ directory
        - classes.txt file
        - Matching image/label file pairs
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        images_dir = self.path / "images"
        labels_dir = self.path / "labels"
        classes_file = self.path / "classes.txt"
        
        # Check required directories
        if not images_dir.exists():
            return False, f"Missing 'images' directory in '{self.path}'."
        
        if not labels_dir.exists():
            return False, f"Missing 'labels' directory in '{self.path}'."
        
        if not classes_file.exists():
            return False, f"Missing 'classes.txt' file in '{self.path}'."
        
        # Check for matching files
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
        missing_images = label_stems - image_stems
        
        if missing_labels:
            return False, f"Missing label files in '{labels_dir}' for {len(missing_labels)} images."
        
        if missing_images:
            return False, f"Missing image files in '{images_dir}' for {len(missing_images)} labels."
        
        return True, ""
    
    def get_classes(self) -> Optional[dict]:
        """Get class names from classes.txt as a dictionary."""
        classes_file = self.path / "classes.txt"
        return parse_classes(classes_file) if classes_file.exists() else None
    
    def get_class_list(self) -> Optional[List[str]]:
        """Get class names as a list."""
        classes = self.get_classes()
        if classes is None:
            return None
        return [classes[i] for i in sorted(classes.keys())]
    
    def get_image_paths(self) -> List[Path]:
        """Get list of image file paths."""
        images_dir = self.path / "images"
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        return [
            f for f in images_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    
    def get_label_path(self, image_path: Path) -> Path:
        """Get the label file path for a given image."""
        return self.path / "labels" / f"{image_path.stem}.txt"

