import json
import os
import uuid
import cv2
import logging
import numpy as np
import shutil
from .colors import COLORS
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
from converter import LS_ROOT_PATH
from urllib.request import (
    pathname2url,
)

log = logging.getLogger("Converter")
default_image_root_url = "/data/local-files/?d=images"

LABELS = """
  <{# TAG_NAME #} name="{# FROM_NAME #}" toName="image">
{# LABELS #}  </{# TAG_NAME #}>
"""

LABELING_CONFIG = """<View>
  <Image name="{# TO_NAME #}" value="$image"/>
{# BODY #}</View>
"""


def generate_label_config(
    categories, tags, to_name="image", from_name="label", filename=None
):
    labels = ""
    for key in sorted(categories.keys()):
        color = COLORS[int(key) % len(COLORS)]
        label = f'    <Label value="{categories[key]}" background="rgba({color[0]}, {color[1]}, {color[2]}, 1)"/>\n'
        labels += label

    body = ""
    for from_name in tags:
        tag_body = (
            str(LABELS)
            .replace("{# TAG_NAME #}", tags[from_name])
            .replace("{# LABELS #}", labels)
            .replace("{# TO_NAME #}", to_name)
            .replace("{# FROM_NAME #}", from_name)
        )
        body += f'\n  <Header value="{tags[from_name]}"/>' + tag_body

    config = (
        str(LABELING_CONFIG)
        .replace("{# BODY #}", body)
        .replace("{# TO_NAME #}", to_name)
    )

    if filename:
        with open(filename, "w") as f:
            f.write(config)

    return config


def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    rle = []
    numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle


def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]


def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):
    """Encode a 1d array to rle


    :param arr: flattened np.array from a 4d image (R, G, B, alpha)
    :type arr: np.array
    :param wordsize: wordsize bits for decoding, default is 8
    :type wordsize: int
    :param rle_sizes:  list of ints which state how long a series is of the same number
    :type rle_sizes: list
    :return rle: run length encoded array
    :type rle: list

    """
    # Set length of array in 32 bits
    num = len(arr)
    numbits = f"{num:032b}"

    # put in the wordsize in bits
    wordsizebits = f"{wordsize - 1:05b}"

    # put rle sizes in the bits
    rle_bits = "".join([f"{x - 1:04b}" for x in rle_sizes])

    # combine it into base string
    base_str = numbits + wordsizebits + rle_bits

    # start with creating the rle bite string
    out_str = ""
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        # TODO: A nice to have but --> this can be optimized but works
        if length_reeks == 1:
            # we state with the first 0 that it has a length of 1
            out_str += "0"
            # We state now the index on the rle sizes
            out_str += "00"

            # the rle size value is 0 for an individual number
            out_str += "000"

            # put the value in a 8 bit string
            out_str += f"{value:08b}"
            state = "single_val"

        elif length_reeks > 1:
            state = "series"
            # rle size = 3
            if length_reeks <= 8:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                # index in rle size arr
                out_str += "00"

                # length of array to bits
                out_str += f"{length_reeks - 1:03b}"

                out_str += f"{value:08b}"

            # rle size = 4
            elif 8 < length_reeks <= 16:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"
                out_str += "01"

                # length of array to bits
                out_str += f"{length_reeks - 1:04b}"

                out_str += f"{value:08b}"

            # rle size = 8
            elif 16 < length_reeks <= 256:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                out_str += "10"

                # length of array to bits
                out_str += f"{length_reeks - 1:08b}"

                out_str += f"{value:08b}"

            # rle size = 16 or longer
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    out_str += "11"
                    out_str += f"{2 ** 16 - 1:016b}"

                    out_str += f"{value:08b}"
                    length_temp -= 2**16

                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                out_str += "11"
                # length of array to bits
                out_str += f"{length_temp - 1:016b}"

                out_str += f"{value:08b}"

    # make sure that we have an 8 fold lenght otherwise add 0's at the end
    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * "0"

    rle = bits2byte(total_str)

    return rle


def mask_to_rle(mask):
    """Convert mask to RLE

    :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
    :return: list of ints in RLE format
    """
    assert len(mask.shape) == 2, "mask must be 2D np.array"
    assert mask.dtype == np.uint8 or mask.dtype == int, "mask must be uint8 or int"
    array = mask.ravel()
    array = np.repeat(array, 4)  # must be 4 channels
    rle = encode_rle(array)
    return rle


def yolo_to_mask(contour: list[float], img_width: int, img_height: int = None) -> np.ndarray:
    """
    Converts a YOLO segmentation mask (polygon format) into a uint8 2D mask.

    :param contour: List of normalized polygon points [x1, y1, x2, y2, ..., xn, yn].
    :param img_width: Original image width.
    :param img_height: Original image height.
    :return: A 2D numpy array (uint8) representing the segmentation mask.
    """

    # Reshape the list into (N,2) coordinates and convert to absolute pixel positions
    polygon = np.array(contour, dtype=np.float32).reshape(-1, 2)
    polygon[:, 0] *= img_width  # Scale X
    polygon[:, 1] *= img_height  # Scale Y
    polygon = polygon.astype(np.int32)  # Convert to integer pixel values

    # Create an empty mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw the polygon as a filled mask
    cv2.fillPoly(img=mask, pts=[polygon], color=[255])  # Fill with white (255)

    return mask


def build_bbox_value(line, categories: dict[int, str],  image_width: int, image_height: int):
    values = line.split()
    label_id, x, y, width, height = values[0:5]

    score = float(values[5]) if len(values) >= 6 else None

    x, y, width, height = (
        float(x),
        float(y),
        float(width),
        float(height),
    )

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


def build_seg_value(line, categories: dict[int, str], image_width: int, image_height: int):
    """Convert mask to brush RLE annotation

        :param path: path to image with mask (jpg, png), this image will be thresholded with values > 128 to obtain mask,
                     so you can mark background as black and foreground as white
        :param label_name: label name from labeling config (<Label>)
        :param from_name: brush tag name (<BrushLabels>)
        :param to_name: image tag name (<Image>)
        :param ground_truth: ground truth annotation true/false
        :param model_version: any string, only for predictions
        :param score: model score as float, only for predictions

        :return: dict with Label Studio Annotation or Prediction (Pre-annotation)
        """
    values = line.split()
    label_id = values[0]
    mask = yolo_to_mask(values[1:], image_width, image_height)
    rle = mask_to_rle(mask)

    item = {
        "id": str(uuid.uuid4())[0:8],
        "type": "brushlabels",
        "value": {"rle": rle, "format": "rle", "brushlabels": [categories[int(label_id)]]},
        "origin": "manual",
    }

    return item


def convert_yolo_to_ls(
    input_dir,
    out_file,
    label_type,
    to_name="image",
    from_name="label",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
    image_dims: Optional[Tuple[int, int]] = None,
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param label_type: string containing the type of the label. Must be either "bbox" or "seg"
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, e.g. .jpeg or .jpg, .png and so on.
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    """

    tasks = []
    log.info("Reading YOLO notes and categories from %s", input_dir)

    # build categories=>labels dict
    notes_file = os.path.join(input_dir, "classes.txt")
    with open(notes_file) as f:
        lines = [line.strip() for line in f.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    log.info(f"Found {len(categories)} categories")

    # generate and save labeling config
    label_config_file = out_file.replace(".json", "") + ".label_config.xml"
    label_type = "RectangleLabels" if label_type == "bbox" else "BrushLabels"
    generate_label_config(
        categories,
        {from_name: label_type},
        to_name,
        from_name,
        label_config_file,
    )

    # define directories
    labels_dir = os.path.join(input_dir, "labels")
    images_dir = os.path.join(input_dir, "images")
    log.info("Converting labels from %s", labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    log.info(f"image extensions->, {image_ext}")

    build_value = build_bbox_value if label_type == "bbox" else build_seg_value

    # loop through images
    for f in os.listdir(images_dir):
        image_file_found_flag = False
        for ext in image_ext:
            if f.endswith(ext):
                image_file = f
                image_file_base = os.path.splitext(f)[0]
                image_file_found_flag = True
                break
        if not image_file_found_flag:
            continue

        image_root_url += "" if image_root_url.endswith("/") else "/"
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url
                + str(pathname2url(image_file))
            }
        }

        # define coresponding label file and check existence
        label_file = os.path.join(labels_dir, image_file_base + ".txt")

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                # default to opening file if we aren't given image dims. slow!
                with Image.open(os.path.join(images_dir, image_file)) as im:
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

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()

                if len(lines) == 0:
                    task[out_type][0]["result"] = []
                else:
                    for line in lines:
                        item = build_value(line, categories, image_width, image_height)
                        item = {**item, **info}
                        task[out_type][0]["result"].append(item)

        tasks.append(task)

    if len(tasks) > 0:
        log.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)

        help_root_dir = ""
        if image_root_url == default_image_root_url:
            help_root_dir = (
                f"Set environment variables LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true and "
                f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={input_dir} for Label Studio run,\n"
                f"add Local Storage with Absolute local path = {input_dir}/images"
            )

        print(
            "\n"
            f"  1. Create a new project in Label Studio\n"
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f"  3. Setup serving for images\n"
            f"       E.g. you can use Local Storage (or others):\n"
            f"       https://labelstud.io/guide/storage.html#Local-storage\n"
            f"       See tutorial here:\nhttps://github.com/HumanSignal/label-studio-converter/tree/master?tab=readme-ov-file#yolo-to-label-studio-converter\n"
            f"       {help_root_dir}\n"
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        log.error("No labels converted")


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


def yolo_to_ls(dataset_dir, dataset_name, label_type):
    """
    Convert a YOLO-formatted dataset to a Label Studio (LS) format.

    Parameters:
      dataset_dir (str): Path to the YOLO dataset.
      dataset_name (str): Base name for the new LS dataset.
      label_type (str): Type of labels, e.g. "bbox" or "seg".
                        Must be either "bbox" or "seg".
    """
    if label_type not in ("bbox", "seg"):
        raise ValueError("label_type must be either 'bbox' or 'seg'")

    # Construct paths based on the label type.
    ls_path = f"{LS_ROOT_PATH}/{dataset_name}"
    image_root = f"/data/local-files/?d={dataset_name}/images"
    output_file = f"{ls_path}/task.json"

    # Create the destination directory.
    os.makedirs(ls_path, exist_ok=True)

    # Copy the "images" directory.
    src_images = os.path.join(dataset_dir, "images")
    dst_images = os.path.join(ls_path, "images")
    shutil.copytree(src_images, dst_images, dirs_exist_ok=True)

    # Copy the "classes.txt" file.
    src_classes = os.path.join(dataset_dir, "classes.txt")
    dst_classes = os.path.join(ls_path, "classes.txt")
    shutil.copy(src_classes, dst_classes)

    # Convert YOLO format to LS using the shared conversion function.
    convert_yolo_to_ls(dataset_dir, output_file, label_type, image_root_url=image_root)


def seg_yolo_to_bbox_yolo():
    pass
