import json
import os
import cv2
import logging
import numpy as np
import shutil
from pathlib import Path
from converter import LS_ROOT_PATH
from label_studio_sdk.converter.imports import yolo

log = logging.getLogger("Converter")


def save_yolo_file(file_name, output_dir, data):
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_path = Path(output_dir) / f"{file_name}.txt"
        with open(output_path, "w", encoding="utf-8") as file:
            for item in data:
                line = " ".join(map(str, item))
                file.write(line + "\n")

        return output_path
    return None


def seg_to_bbox(seg_info: list):
    """
    Converts a segmentation label in YOLO format to a bounding box (bbox) label.

    Parameters:
        seg_info (list): A list containing segmentation information.
                         - The first element is the class ID.
                         - The remaining elements are pairs of (x, y) coordinates defining the polygon.

    Returns:
        list: A bounding box representation in YOLO format as:
              [class_id, x_center, y_center, width, height]
              - class_id (int): The zero-based class index.
              - x_center (float): The normalized x-coordinate of the bbox center.
              - y_center (float): The normalized y-coordinate of the bbox center.
              - width (float): The normalized width of the bbox.
              - height (float): The normalized height of the bbox.
    """

    # Extract the class ID (first element) and the list of points (remaining elements).
    class_id, *points = seg_info

    # Convert all point coordinates from string to float
    points = [float(p) for p in points]

    # Compute the bounding box (bbox) from the polygon points
    x_min, y_min = min(points[0::2]), min(points[1::2])  # Smallest x and y values
    x_max, y_max = max(points[0::2]), max(points[1::2])  # Largest x and y values

    # Compute width and height of the bbox
    width, height = x_max - x_min, y_max - y_min

    # Compute the center coordinates of the bbox
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    # Convert class ID from 1-based to 0-based (if needed)
    bbox_info = [int(class_id) - 1, x_center, y_center, width, height]

    return bbox_info


def binmask_to_yolo(binmask_path, output_seg_path=None, output_box_path=None):
    """
    Converts a dataset of binary segmentation mask images to the YOLO format.

    This function takes the directory containing the binary format mask images and converts them into YOLO format.
    The converted files are saved in the specified output directories.

    Args:
        binmask_path (str): The path to the directory where all mask images (png, jpg) are stored.
        output_seg_path (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        output_box_path (str): The path to the directory where the converted YOLO bounding boxes will be stored.

    Notes:
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_seg_dir
                ├─ mask_image_01_seg.txt
                ├─ mask_image_02_seg.txt
                ├─ mask_image_03_seg.txt
                └─ mask_image_04_seg.txt

            - output_box_dir
                ├─ mask_image_01_box.txt
                ├─ mask_image_02_box.txt
                ├─ mask_image_03_box.txt
                └─ mask_image_04_box.txt
    """

    for file_path in Path(binmask_path).iterdir():
        if file_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            print(f"Processing {file_path} imgsz = {img_height} x {img_width}")

            # Create a binary mask for the current class and find contours
            contours, _ = cv2.findContours(
                (mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours

            seg_info_list = []
            bbox_info_list = []

            for contour in contours:
                if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                    contour = contour.squeeze()  # Remove single-dimensional entries

                    seg_info = [1]
                    for point in contour:
                        # Normalize the coordinates
                        seg_info.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                        seg_info.append(round(point[1] / img_height, 6))

                    bbox_info = seg_to_bbox(seg_info)

                    seg_info_list.append(seg_info)
                    bbox_info_list.append(bbox_info)

            out_name = file_path.stem.replace("_mask", "")

            res_path = save_yolo_file(out_name, output_seg_path, seg_info_list)
            if res_path is not None:
                print(f"Processed and stored binary segmentation map at {res_path} imgsz = {img_height} x {img_width}")
            else:
                print(f"There was an error trying to save the segmentation map from {file_path.stem}.")

            res_path = save_yolo_file(out_name, output_box_path, bbox_info_list)
            if res_path is not None:
                print(f"Processed and stored bounding boxes at {res_path} imgsz = {img_height} x {img_width}")
            else:
                print(f"There was an error trying to save the bounding boxes from {file_path.stem}.")


def ls_to_yolo(ls_data_name, output_dir, label_type):
    """
    Converts an LS dataset into the YOLO format.

    This function takes the directory containing the images and a .json task file and converts them into YOLO format.
    The converted files are saved in the specified output directories.

    NOTE: to allow for flexibility in the .json file name, the function looks for the first .json file in the directory.
    Any other .json file will be ignored.

    Args:
        ls_data_name (str): The name of the directory where the Label Studio dataset is stored.
        output_dir (str): The path to the directory where the converted YOLO labels will be stored.
        label_type (str): Type of labels. Must be either "bbox" or "seg".

    Notes:
        The expected directory structure for the masks is:

        - [Label Studio Root]
            - <ls_data_name>
                ├─ images
                ├─ ...
                └─ task.json
    """
    ls_data_path = LS_ROOT_PATH / ls_data_name

    if label_type not in ("bbox", "seg"):
        log.error(f"Label type {label_type} not supported.")

    if not ls_data_path.exists():
        log.error("The selected Label Studio dataset does not exist")
        return False

    images_path = ls_data_path / "images"
    json_path = next(ls_data_path.glob("*.json"))

    if not images_path.exists() or not json_path.exists():
        log.error(f"The selected Label Studio dataset does not contain the required files. ({ls_data_path})")
        return False

    if os.path.exists(output_dir):
        log.warning(f"The selected output directory already exists. ({output_dir})")

    output_path = Path(output_dir)
    log.info(f"Creating output directory: {output_path}")
    output_path.mkdir(parents=True)

    labels_path = output_path / "labels"
    labels_path.mkdir()

    task_file = open(json_path, "r", encoding="utf-8")
    tasks = json.load(task_file)
    task_file.close()

    for task in tasks:
        image_filename = task["data"]["image"].split("/")[-1]
        image_path = images_path / image_filename

        if not image_path.exists():
            log.warning(f"Image file not found: {image_path}")
            continue

        log.info(f"Processing annotations for image: {image_path}")

        label_filename = image_filename.replace('.jpg', '.txt')
        label_path = labels_path / label_filename

        annotation_lines = []

        for annotation in task["annotations"]:
            for result in annotation["result"]:
                value = result["value"]
                annotation_class = 0
                annotation_info = []

                if label_type == "bbox":
                    annotation_info = [annotation_class, value["x"], value["y"], value["width"], value["height"]]
                elif label_type == "label":
                    annotation_info = [annotation_class, *value["rle"]]

                annotation_lines.append(' '.join(annotation_info))

        label_file = label_path.open("w", encoding="utf-8")
        label_file.writelines(annotation_lines)
        label_file.close()

    classes_path = output_path / "classes.txt"
    classes_file = open(classes_path, "w", encoding="utf-8")
    classes_file.write("Foam")
    classes_file.close()

    return True


def yolo_to_ls(yolo_dir, ls_dir_name, label_type):
    """
    Convert a YOLO-formatted dataset to a Label Studio (LS) format.

    Parameters:
      yolo_path (str or Path): Path to the YOLO dataset.
      yolo_dir_name (str): Base name for the new LS dataset.
      label_type (str): Type of labels, e.g. "bbox" or "seg".
                        Must be either "bbox" or "seg".
    """
    if label_type not in ("bbox", "seg"):
        raise ValueError("label_type must be either 'bbox' or 'seg'")

    yolo_path = Path(yolo_dir)
    ls_path = Path(LS_ROOT_PATH) / ls_dir_name
    image_root = f"/data/local-files/?d={ls_dir_name}/images"
    output_file = ls_path / "task.json"

    ls_path.mkdir(parents=True, exist_ok=True)

    src_images = yolo_path / "images"
    dst_images = ls_path / "images"
    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

    src_classes = yolo_path / "classes.txt"
    dst_classes = ls_path / "classes.txt"
    shutil.copy(src_classes, dst_classes)

    yolo.convert_yolo_to_ls(str(yolo_path), str(output_file), image_root_url=image_root)


def seg_yolo_to_bbox_yolo(yolo_root_dir, yolo_seg_dir_name):
    """
    Converts a YOLO dataset with segmentation labels into a YOLO dataset with bounding box labels.

    The function expects the YOLO segmentation directory to have the following structure:
        YOLO SEG FORMAT:
            images/       --> images in jpg or png format
            labels/       --> text files with segmentation labels (each line: <class_id> <x1> <y1> <x2> <y2> ... )
            classes.txt  --> list of classes where each line corresponds to a class (order defines the index)

    It creates a new YOLO dataset with bounding box labels in the same parent directory.
    The new directory name is derived by replacing "YOLOseg" with "YOLObbox" in the original directory name.

    For each label file in the segmentation dataset, the function:
      - Reads each line.
      - Splits the line into tokens.
      - Converts the segmentation (polygon) data into a bounding box using the seg_to_bbox() function.
      - Writes the new bounding box info (formatted as: <class_id> <x_center> <y_center> <width> <height>)
        to a corresponding label file in the new YOLObbox dataset.

    Parameters:
        yolo_root_path (str or Path): Path to the parent directory containing the YOLO dataset directories.
        yolo_seg_dir_name (str): Name of the YOLO segmentation directory (e.g., "Xylella_YOLOseg").

    Output:
        A new directory is created (e.g., "Xylella_YOLObbox") with:
          - images/ (copied from the segmentation dataset)
          - labels/ (converted from segmentation to bounding box labels)
          - classes.txt (copied from the segmentation dataset)
    """
    yolo_root_path = Path(yolo_root_dir)

    seg_dir = yolo_root_path / yolo_seg_dir_name

    yolo_bbox_dir_name = yolo_seg_dir_name.replace("YOLOseg", "YOLObbox")
    bbox_dir = yolo_root_path / yolo_bbox_dir_name

    # (bbox_dir / "images").mkdir(parents=True, exist_ok=True)
    (bbox_dir / "labels").mkdir(parents=True, exist_ok=True)

    src_images = seg_dir / "images"
    dst_images = bbox_dir / "images"
    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

    src_classes = seg_dir / "classes.txt"
    dst_classes = bbox_dir / "classes.txt"
    shutil.copy(src_classes, dst_classes)

    src_labels_dir = seg_dir / "labels"
    dst_labels_dir = bbox_dir / "labels"

    # Process each label file in the segmentation dataset.
    for label_file in src_labels_dir.iterdir():
        if not label_file.is_file():
            continue

        dst_label_path = dst_labels_dir / label_file.name

        with label_file.open("r") as f:
            seg_lines = f.readlines()

        bbox_lines = []

        for seg_line in seg_lines:
            seg_line = seg_line.strip()

            if not seg_line:
                continue

            seg_info = seg_line.split()
            bbox_info = seg_to_bbox(seg_info)
            bbox_line = " ".join(str(x) for x in bbox_info)
            bbox_lines.append(bbox_line)

        with dst_label_path.open("w") as f:
            f.writelines(bbox_lines)
