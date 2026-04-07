"""
YOLO to Ultralytics format conversion.

Converts standard YOLO format (images/, labels/, classes.txt) to
Ultralytics YOLO format (train/val/test splits with .yaml config).
"""

import random
import shutil
import logging
import yaml
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
from cvtoolkit.formats.format import FormatType
from cvtoolkit.conversions.conversion import Conversion, register_conversion
from cvtoolkit.formats.yolo import parse_classes


log = logging.getLogger("YoloToUl")


def build_filenames_list(
    images_path: Path, 
    labels_path: Path, 
    image_ext: List[str]
) -> List[str]:
    """
    Build a list of base filenames that have both image and label files.
    
    Args:
        images_path: Path to the directory containing image files
        labels_path: Path to the directory containing label files
        image_ext: List of valid image file extensions
    
    Returns:
        List of base filenames (without extensions)
    """
    all_files = []
    
    for img_path in images_path.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in image_ext:
            base_name = img_path.stem
            label_path = labels_path / f"{base_name}.txt"
            if label_path.is_file():
                all_files.append(base_name)
            else:
                log.warning(f"Label file missing for image: {img_path.name}")
    
    if not all_files:
        log.error("No valid image/label pairs found")
    else:
        log.info(f"Found {len(all_files)} matching image/label pairs")
    
    return all_files


def shuffle_and_split(
    input_list: List[str],
    split_ratios: Tuple[float, float, Optional[float]],
    include_test_split: bool = False
) -> dict | None:
    """
    Shuffle and split a list of filenames into train/val/test sets.
    
    Args:
        input_list: List of filenames to shuffle and split
        split_ratios: Tuple of ratios (train, val, [test])
        include_test_split: Whether to create a test split
    
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' keys
    """
    random.shuffle(input_list)
    total_files = len(input_list)
    train_ratio, val_ratio = split_ratios[:2]
    test_ratio = split_ratios[2] if include_test_split and len(split_ratios) > 2 else 0.0

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio < 0:
        log.error(f"Invalid split ratios: {split_ratios}")
        return None

    # Normalize ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        log.warning(f"Split ratios do not sum to 1. Normalizing.")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    n_train = int(total_files * train_ratio)

    if include_test_split:
        n_val = int(total_files * val_ratio)
        n_test = total_files - n_train - n_val
    else:
        n_val = total_files - n_train
        n_test = 0

    split_data = {
        "train": input_list[:n_train],
        "val": input_list[n_train: n_train + n_val],
    }

    if include_test_split and n_test > 0:
        split_data["test"] = input_list[n_train + n_val:]
        log.info(f"Splitting data: {n_train} train, {n_val} val, {n_test} test")
    else:
        log.info(f"Splitting data: {n_train} train, {n_val} val")

    return split_data


@register_conversion(FormatType.YOLO, FormatType.ULTRALYTICS)
class YoloToUltralytics(Conversion):
    """Base class for YOLO to Ultralytics conversions."""
    
    def convert(
        self,
        path_in_yaml: str,
        split_ratios: Tuple[float, float, Optional[float]] = (0.8, 0.2),
        include_test_split: bool = False,
        image_ext: str = ".jpg,.png",
        yaml_filename: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Path:
        """
        Convert YOLO dataset to Ultralytics format.
        
        Args:
            path_in_yaml: Path to the dataset in the YAML file
            split_ratios: Train/val/(test) split ratios
            include_test_split: Whether to create a test split
            image_ext: Comma-separated list of image extensions
            yaml_filename: Name for the output YAML file
            random_seed: Seed for reproducible splits
        
        Returns:
            Path to the Ultralytics dataset
        """
        image_ext_list = [ext.strip().lower() for ext in image_ext.split(",")]
        
        yolo_images = self.source_path / "images"
        yolo_labels = self.source_path / "labels"
        classes_file = self.source_path / "classes.txt"
        
        # Validate source structure
        if not yolo_images.is_dir():
            raise ValueError(f"Source images directory not found: {yolo_images}")
        if not yolo_labels.is_dir():
            raise ValueError(f"Source labels directory not found: {yolo_labels}")
        if not classes_file.is_file():
            raise ValueError(f"Classes file not found: {classes_file}")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        self._report_progress(0.25, "Building file list & splitting...")
        
        # Create target directory
        self.target_path.mkdir(parents=True, exist_ok=True)
        self._track_path(self.target_path)
        
        classes_dict = parse_classes(classes_file)
        all_files = build_filenames_list(yolo_images, yolo_labels, image_ext_list)
        
        if not all_files:
            raise ValueError("No valid image/label pairs found")
        
        split_data = shuffle_and_split(all_files, split_ratios, include_test_split)
        
        if split_data is None:
            raise ValueError("Failed to split data")
        
        yaml_data = {
            'path': path_in_yaml,
        }
        
        # Distribute progress across splits (0.30 to 0.90)
        num_splits = len(split_data)
        split_progress_range = 0.60  # 0.30 to 0.90
        
        for split_idx, (split_name, file_list) in enumerate(split_data.items()):
            split_start = 0.30 + (split_progress_range * split_idx / num_splits)
            split_end = 0.30 + (split_progress_range * (split_idx + 1) / num_splits)
            
            img_dest = self.target_path / split_name / "images"
            lbl_dest = self.target_path / split_name / "labels"
            img_dest.mkdir(parents=True, exist_ok=True)
            lbl_dest.mkdir(parents=True, exist_ok=True)
            
            yaml_data[split_name] = f"{split_name}/images"
            
            log.info(f"Copying {len(file_list)} files to {split_name}...")
            total_files = len(file_list)
            
            for i, base_name in enumerate(tqdm(file_list, ascii="░▒█", desc=f"Creating {split_name}")):
                # Find the original image file
                original_img = None
                for img_path in yolo_images.glob(f"{base_name}.*"):
                    if img_path.suffix.lower() in image_ext_list:
                        original_img = img_path
                        break
                
                if original_img:
                    original_lbl = yolo_labels / f"{base_name}.txt"
                    dest_img = img_dest / original_img.name
                    dest_lbl = lbl_dest / f"{base_name}.txt"
                    
                    try:
                        shutil.copy2(original_img, dest_img)
                        shutil.copy2(original_lbl, dest_lbl)
                    except Exception as e:
                        log.error(f"Error copying {base_name}: {e}")
                
                # Report per-file progress within this split's range
                progress = split_start + (split_end - split_start) * (i + 1) / total_files
                self._report_progress(progress, f"Copying {split_name} {i + 1}/{total_files}")
        
        self._report_progress(0.95, "Writing YAML config...")
        
        yaml_data['names'] = classes_dict
        
        if yaml_filename is None:
            yaml_filename = f"{self.target_path.name}.yaml"
        
        yaml_path = self.target_path / yaml_filename
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
        
        log.info(f"Created YAML config: {yaml_path}")
        log.info(f"Conversion complete: {self.target_path}")
        
        return self.target_path

