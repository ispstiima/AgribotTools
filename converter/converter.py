import json
import uuid
import cv2
import logging
import numpy as np
import shutil
from .colors import COLORS
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple
from urllib.request import pathname2url
import os


log = logging.getLogger("Converter")
logging.basicConfig(filename="converter.log", level=logging.INFO, format="%(asctime)s %(message)s",)

default_image_root_url = "/data/local-files/?d=images"

LABELS = """
  <{# TAG_NAME #} name="{# FROM_NAME #}" toName="image">
{# LABELS #}  </{# TAG_NAME #}>
"""

LABELING_CONFIG = """<View>
  <Image name="{# TO_NAME #}" value="$image"/>
{# BODY #}</View>
"""

LS_ROOT = Path(os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"])
OUT_DIR = "./out"

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)


def generate_label_config(config_path, categories, tags, to_name="image"):
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

    config_path.write_text(config)
    log.info(f"Label configuration file saved to: {config_path}")

    return config


def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :rtype: list
    """
    rle = []
    numbers = [arr_str[i: i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle


def bytes2bit(data):
    """get bit string from bytes data"""
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def base_rle_encode(inarray):
    """
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)
    """
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


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):
    """Encode a 1d array to rle


    :param arr: flattened np.array from a 4d image (R, G, B, alpha)
    :type arr: np.array
    :param wordsize: wordsize bits for decoding, default is 8
    :type wordsize: int
    :param rle_sizes:  list of ints which state how long a series is of the same number
    :type rle_sizes: list
    :return rle: run length encoded array
    :rtype: list

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
                while length_temp > 2 ** 16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    out_str += "11"
                    out_str += f"{2 ** 16 - 1:016b}"

                    out_str += f"{value:08b}"
                    length_temp -= 2 ** 16

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


def decode_rle(rle):
    """
    Decode an RLE-Encoded list of integers to a flattened numpy uint8 image.
    [width, height, channel]


    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out


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


def mask_to_yolo(mask: np.ndarray, seg: bool = True, bbox: bool = False):
    # TODO add the logic introduced by this function into binmask_to_yolo()
    img_height, img_width = mask.shape

    contours, _ = cv2.findContours(
        (mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    seg_yolo = [] if seg else None
    bbox_yolo = [] if bbox else None

    for contour in contours:
        if len(contour) >= 3:
            contour = contour.squeeze()

            seg_line = []
            for point in contour:
                seg_line.append(round(point[0] / img_width, 6))
                seg_line.append(round(point[1] / img_height, 6))

            if seg:
                seg_yolo.append(seg_line)

            if bbox:
                bbox_yolo.append(seg_to_bbox(seg_line))

    print(f'seg_yolo: {seg_yolo}')
    return seg_yolo, bbox_yolo


def build_bbox_value(line, categories: dict[int, str], image_width: int, image_height: int):
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


def parse_bbox_value(value, img_w, img_h):
    """
    Convert LS annotation to Yolo format.

    Args:
        value (dict): Dictionary containing annotation information including:
            - width (float): Width of the object.
            - height (float): Height of the object.
            - x (float): X-coordinate of the top-left corner of the object.
            - y (float): Y-coordinate of the top-left corner of the object.

    Returns:
        tuple or None: If the conversion is successful, returns a tuple (x, y, w, h) representing
        the coordinates and dimensions of the object in Yolo format, where (x, y) are the center
        coordinates of the object, and (w, h) are the width and height of the object respectively.
    """

    if not ("x" in value and "y" in value and "width" in value and "height" in value):
        return None

    w = value["width"]
    h = value["height"]

    x = (value["x"] + w / 2) / 100
    y = (value["y"] + h / 2) / 100
    w = w / 100
    h = h / 100

    return x, y, w, h


def parse_seg_value(value, img_w, img_h):
    """
    Convert Segmentation Mask LS annotation to Yolo format.

    Args:
        value (dict): Dictionary containing annotation information including:
            - rle (list): RLE-Encoded list of integers representing the segmentation mask.
        img_w (int): Image width
        img_h (int): Image height
    Returns:
        tuple or None: If the conversion is successful, returns an ndarray containing the segmentation mask.
    """
    flat_binmask = decode_rle(value["rle"])
    binmask = np.reshape(flat_binmask, [img_h, img_w, 4])[:, :, 3]
    seg_yolo, _ = mask_to_yolo(binmask)
    return seg_yolo[0]  # Extracting segmentation masks from the LS task should return just one mask

def validate_dataset(images_dir_path, labels_dir_path):
    assert os.path.exists(images_dir_path), f"Path {images_dir_path} does not exist"
    assert os.path.exists(labels_dir_path), f"Path {labels_dir_path} does not exist"
    
    image_files = {os.path.splitext(file)[0] for file in os.listdir(images_dir_path) if os.path.isfile(os.path.join(images_dir_path, file))}
    binmask_files = {os.path.splitext(file)[0] for file in os.listdir(labels_dir_path) if os.path.isfile(os.path.join(labels_dir_path, file))}
    
    missing_in_labels = image_files - binmask_files
    missing_in_images = binmask_files - image_files
    
    if missing_in_labels:
        raise ValueError(f"The following files are missing in labels_dir_path: {missing_in_labels}")
    if missing_in_images:
        raise ValueError(f"The following files are missing in images_dir_path: {missing_in_images}")
    
    return True

def save_yolo_txt_file(file_name, output_dir, data):
    assert data, "Data is empty"
    assert file_name, "File name is empty"
    assert isinstance(data, list), "Data should be a list"
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f"{file_name}.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for item in data:
            line = " ".join(map(str, item))
            file.write(line + "\n")


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


def binmask_to_yolo(dataset_path, should_make_seg, should_make_bbox):
    """
    Converts binary segmentation mask images to YOLO format for segmentation and bounding boxes annotations types.

    This function processes binary mask images and generates YOLO-compatible labels for segmentation 
    and/or bounding boxes annotations types.
        
        Args:
            dataset_path (str): The path to the directory containing the dataset with subdirectory containing images (png or jpg format) and labels containing binary masks.
            should_make_seg (bool): Flag to indicate whether YOLO segmentation labels should be generated.
            should_make_bbox (bool): Flag to indicate whether YOLO bounding box labels should be generated.
            
        Notes:
            - At least one of the flags should be set to True.
            - The binary mask images should have pixel values of 255 for the object and 0 for the background.
            - The function normalizes coordinates to the range [0, 1] for YOLO format.

    Directory Structures:
        Input:
            dataset_path/
                |
                ├─ images/
                |   ├─ image_01.png
                |   ├─ image_02.png
                |   ├─ image_03.png
                |   └─ image_04.png
                |    
                ├─ labels/
                |   ├─ image_01.png
                |   ├─ image_02.png
                |   ├─ image_03.png
                |   └─ image_04.png
                |
                └─ classes.txt
                

        Output:
            out/
                │ 
                └─ <dataset_path>_yolo_<annotation_type>/
                    |
                    ├─ images/
                    │   ├─ image_01.png
                    │   ├─ image_02.png
                    │   ├─ image_03.png
                    │   └─ image_04.png
                    |
                    ├─ labels/
                    |   ├─ image_01.txt
                    |   ├─ image_02.txt
                    |   ├─ image_03.txt
                    |   └─ image_04.txt
                    |
                    └─ classes.txt 
    Returns:
        None

    Example:
        binmask_to_yolo(
            dataset_path="/path/to/dataset",
            should_make_seg=True,
            should_make_bbox=True
        )
    """
    
    normalized_dataset_path = os.path.normpath(dataset_path)
    
    images_dir_path = f"{normalized_dataset_path}/images"
    binmasks_dir_path = f"{normalized_dataset_path}/labels"
    validate_dataset(images_dir_path, binmasks_dir_path)
    
    assert should_make_seg or should_make_bbox, "At least one of the flags --seg --bbox should be set to True"
    assert not(should_make_bbox and should_make_seg), "Both flags --seg and --bbox cannot be set to True at the same time"
    
    dataset_name = os.path.basename(normalized_dataset_path)
    yolo_dataset_path = f"{OUT_DIR}/{dataset_name}"
    
    if should_make_seg:
        yolo_dataset_path = f"{yolo_dataset_path}_yolo_seg"
    elif should_make_bbox:
        yolo_dataset_path = f"{yolo_dataset_path}_yolo_bbox"
    yolo_images_path_dir = f"{yolo_dataset_path}/images"
    yolo_labels_path_dir = f"{yolo_dataset_path}/labels"
    os.makedirs(yolo_images_path_dir, exist_ok=True)
    os.makedirs(yolo_labels_path_dir, exist_ok=True)

    shutil.copytree(images_dir_path, yolo_images_path_dir, dirs_exist_ok=True)
    shutil.copy(f"{normalized_dataset_path}/classes.txt", f"{yolo_dataset_path}/classes.txt")

    for binmask_file_path in Path(binmasks_dir_path).iterdir():
        assert binmask_file_path.suffix in [".png", ".jpg"], f"Unsupported file format: {binmask_file_path.suffix}"
        print(f"Converting labels from {binmask_file_path} \n")
        mask = cv2.imread(str(binmask_file_path), cv2.IMREAD_GRAYSCALE)  # binmask_Read the mask image in grayscale
        seg_list, bbox_list = mask_to_yolo(mask, should_make_seg, should_make_bbox)            
        if should_make_seg:
            label_list = seg_list
        if should_make_bbox:
            label_list = bbox_list
        save_yolo_txt_file(binmask_file_path.stem, yolo_labels_path_dir, label_list)
        

def convert_yolo_to_ls(
        yolo_path,
        json_path,
        label_type,
        to_name="image",
        from_name="label",
        out_type="annotations",
        image_root_url=default_image_root_url,
        image_ext=".jpg,.png",
        image_dims: Optional[Tuple[int, int]] = None,
):
    """Convert YOLO labeling to Label Studio JSON
    :param yolo_path: path where images, labels, notes.json are located
    :param json_path: output file with Label Studio JSON tasks
    :param label_type: string containing the type of the label. Must be either "bbox" or "seg"
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, e.g. .jpeg or .jpg, .png and so on.
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    """

    tasks = []
    log.info(f"> Reading YOLO files from: [{yolo_path}]")

    # build categories=>labels dict
    classes_path = yolo_path / "classes.txt"
    with classes_path.open() as classes_file:
        lines = [line.strip() for line in classes_file.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    log.info(f"Found {len(categories)} categories")

    # generate and save labeling config
    config_path = json_path.parent / "template.label_config.xml"
    label_tag = "RectangleLabels" if label_type == "bbox" else "BrushLabels"

    generate_label_config(
        config_path,
        categories,
        {from_name: label_tag},
        to_name,
    )

    # define directories
    labels_path = yolo_path / "labels"
    images_path = yolo_path / "images"
    log.info(f"Converting labels from {labels_path}")

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    log.info(f"image extensions->, {image_ext}")

    if label_type == "bbox":
        build_value = build_bbox_value
    elif label_type == "seg":
        build_value = build_seg_value
    else:
        log.error(f"The specified label type '{label_type}' is not supported. Please use 'bbox' or 'seg'.")
        return

    # loop through images
    for image_path in images_path.iterdir():
        if not image_path.suffix in image_ext:
            log.warning(f"The image '{image_path}' has an unsupported extension. Skipping.")
            continue

        image_root_url += "" if image_root_url.endswith("/") else "/"
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url + pathname2url(image_path.name)
            }
        }

        label_path = labels_path / f"{image_path.stem}.txt"

        if label_path.exists():
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                # default to opening file if we aren't given image dims. slow!
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

            with label_path.open("r") as file:
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
        log.info(f"Saving Label Studio JSON to {json_path}")
        with json_path.open("w") as out:
            json.dump(tasks, out)

        help_root_dir = ""
        if image_root_url == default_image_root_url:
            help_root_dir = (
                f"Set environment variables LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true and "
                f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={yolo_path} for Label Studio run,\n"
                f"add Local Storage with Absolute local path = {yolo_path}/images"
            )

        print(
            "\n"
            f"  1. Create a new project in Label Studio\n"
            f'  2. Use Labeling Config from "{config_path.name}"\n'
            f"  3. Setup serving for images\n"
            f"       E.g. you can use Local Storage (or others):\n"
            f"       https://labelstud.io/guide/storage.html#Local-storage\n"
            f"       See tutorial here:\nhttps://github.com/HumanSignal/label-studio-converter/tree/master?tab=readme-ov-file#yolo-to-label-studio-converter\n"
            f"       {help_root_dir}\n"
            f'  4. Import "{json_path.name}" to the project\n'
        )
    else:
        log.error("No labels converted")


def convert_ls_to_yolo(ls_path: Path, yolo_path: Path, label_type: str, image_ext=".jpg,.png") -> bool:
    """
    Converts an LS dataset into the YOLO format.

    This function takes the directory containing the images and a .json task file and converts them into YOLO format.
    The converted files are saved in the specified output directories.

    NOTE: to allow for flexibility in the .json file name, the function looks for the first .json file in the directory.
    Any other .json file will be ignored.

    Args:
        ls_path (path): Path to Label Studio dataset.
        yolo_path (path): Path to the directory where the converted YOLO labels will be stored.
        label_type (str): Type of labels. Must be either "bbox" or "seg".
    """

    if label_type not in ("bbox", "seg"):
        log.error(f"Label type {label_type} not supported.")

    if not ls_path.exists():
        log.error("The selected Label Studio dataset does not exist")
        return False

    images_path = ls_path / "images"
    json_path = next(ls_path.glob("*.json"))

    if not images_path.exists() or not json_path.exists():
        log.error(f"The selected Label Studio dataset does not contain the required files. ({ls_path})")
        return False

    labels_path = yolo_path / "labels"
    labels_path.mkdir(exist_ok=True)

    with json_path.open("r", encoding="utf-8") as task_file:
        tasks = json.load(task_file)

    if tasks is None:
        log.error("No tasks found in the task file.")
        return False

    for task in tasks:
        image_filename: str = task["data"]["image"].split("/")[-1]
        image_path = images_path / image_filename

        if not image_path.exists():
            log.warning(f"Image file not found: {image_path}")
            continue

        log.info(f"Processing annotations for image: {image_path}")

        label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = labels_path / label_filename

        parse_value = parse_bbox_value if label_type == "bbox" else parse_seg_value

        print(f">> Processing task of image {image_filename}")

        yolo_lines = []

        for annotation in task["annotations"]:
            print(f"> Result")
            for result in annotation["result"]:
                img_w, img_h = result["original_width"], result["original_height"]
                value = result["value"]
                yolo_class = 0  # TODO allow multiclass convertion

                yolo_line = [yolo_class, *parse_value(value, img_w, img_h)]

                print("Line: " + str(yolo_line[:10]))
                yolo_lines.append(' '.join([str(x) for x in yolo_line]))

        print(f"YOLO File: {yolo_lines}")

        label_file = label_path.open("w", encoding="utf-8")
        for line in yolo_lines:
            label_file.write(line)
            label_file.write("\n")
        # label_file.writelines(yolo_lines)
        label_file.close()

    classes_path = yolo_path / "classes.txt"
    classes_file = classes_path.open(mode="w", encoding="utf-8")
    classes_file.write("Foam")
    classes_file.close()

    return True


def yolo_to_ls(yolo_dir: str, ls_base_name: str, label_type: str, reverse=False):
    """
    Convert a YOLO-formatted dataset to a Label Studio (LS) format.

    Parameters:
      yolo_dir (str): Path to the YOLO dataset.
      ls_base_name (str): Base name for the new LS dataset.
      label_type (str): Type of labels, e.g. "bbox" or "seg".
                        Must be either "bbox" or "seg".
      reverse (bool): If True, the function will reverse the conversion.
    """

    yolo_path = Path(yolo_dir)
    ls_path = LS_ROOT_PATH / ls_base_name

    if label_type not in ("bbox", "seg"):
        raise ValueError("label_type must be either 'bbox' or 'seg'")

    yolo_images = yolo_path / "images"
    yolo_classes = yolo_path / "classes.txt"
    ls_images = ls_path / "images"
    ls_classes = ls_path / "classes.txt"

    if reverse:
        yolo_path.mkdir(parents=True, exist_ok=True)

        convert_ls_to_yolo(ls_path, yolo_path, label_type)

        shutil.copytree(ls_images, yolo_images, dirs_exist_ok=True)
        shutil.copy(ls_classes, yolo_classes)
    else:
        image_root = f"/data/local-files/?d={ls_base_name}/images"
        json_path = ls_path / "task.json"
        ls_path.mkdir(parents=True, exist_ok=True)

        convert_yolo_to_ls(yolo_path, json_path, label_type, image_root_url=image_root)

        shutil.copytree(yolo_images, ls_images, dirs_exist_ok=True)
        shutil.copy(yolo_classes, ls_classes)


def seg_yolo_to_bbox_yolo():
    pass
