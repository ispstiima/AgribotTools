import cv2
from pathlib import Path
import numpy as np


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
    # Example input: 5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...
    class_id, *points = seg_info
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    bbox_info = [int(class_id)-1, x_center, y_center, width, height]
    return bbox_info


def convert_binary_masks_to_yolo(masks_dir, output_seg_dir=None, output_box_dir=None):
    """
    Converts a dataset of binary segmentation mask images to the YOLO format.

    This function takes the directory containing the binary format mask images and converts them into YOLO format.
    The converted files are saved in the specified output directories.

    Args:
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_seg_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        output_box_dir (str): The path to the directory where the converted YOLO bounding boxes will be stored.

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

    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # Read the mask image in grayscale
            img_height, img_width = mask.shape  # Get image dimensions
            print(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

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

            res_path = save_yolo_file(f"{mask_path.stem}_seg", output_seg_dir, seg_info_list)

            if res_path is not None:
                print(f"Processed and stored binary segmentation map at {res_path} imgsz = {img_height} x {img_width}")
            else:
                print(f"There was an error trying to save the segmentation map from {mask_path.stem}.")

            res_path = save_yolo_file(f"{mask_path.stem}_box", output_box_dir, bbox_info_list)

            if res_path is not None:
                print(f"Processed and stored bounding boxes at {res_path} imgsz = {img_height} x {img_width}")
            else:
                print(f"There was an error trying to save the bounding boxes from {mask_path.stem}.")