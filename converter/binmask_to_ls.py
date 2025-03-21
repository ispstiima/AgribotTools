import argparse
import os
from converter.converter import binmask_to_yolo
from label_studio_sdk.converter.imports import yolo


def main():
    parser = argparse.ArgumentParser(description='Convert binary segmentation masks to Label Studio format')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--seg',  '-s', action='store_true')
    parser.add_argument('--bbox', '-b', action='store_true')
    parser.add_argument('--output_seg_path', default="./out/seg", type=str)
    parser.add_argument('--output_box_path', default="./out/box", type=str)

    args = parser.parse_args()

    dataset_dir = args.dataset_path
    dataset_name = dataset_dir.split("/")[-1].replace("/", "")

    yolo_root = "./yolo_temp"
    ls_root = os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]

    yolo_seg_path = f"{yolo_root}/{dataset_name}_seg" if args.seg else None
    yolo_bbox_path = f"{yolo_root}/{dataset_name}_bbox" if args.bbox else None

    binmask_to_yolo(f"{dataset_dir}/", yolo_seg_path, yolo_bbox_path)

    if args.seg:
        ls_seg_path = f"{ls_root}/{dataset_name}_seg"

        # crea cartella ls_seg in ls_root
        # copia images e classes.txt in ls_seg

        image_root = f"/data/local-files/?d={dataset_name}_seg/images"
        output_file = f"{ls_seg_path}/task.json"

        yolo.convert_yolo_to_ls(ls_seg_path, output_file, image_root_url=image_root)

    if args.bbox:
        ls_bbox_path = f"{ls_root}/{dataset_name}_bbox"

        # crea cartella ls_bbox in ls_root
        # copia images e classes.txt in ls_bbox

        image_root = f"/data/local-files/?d={dataset_name}_bbox/images"
        output_file = f"{ls_bbox_path}/task.json"

        yolo.convert_yolo_to_ls(dataset_dir, output_file, image_root_url=image_root)


if __name__ == '__main__':
    main()
