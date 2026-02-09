import argparse
from cvtoolkit.converter import binmask_to_yolo


def main():
    parser = argparse.ArgumentParser(description='Convert binary segmentation masks to Label Studio format')
    parser.add_argument('--seg',  '-s', action='store_true')
    parser.add_argument('--bbox', '-b', action='store_true')
    parser.add_argument('--output_seg_path', default="./out/seg", type=str)
    parser.add_argument('--output_box_path', default="./out/box", type=str)
    parser.add_argument('binmask_path', type=str)

    args = parser.parse_args()

    dataset_dir = args.binmask_path
    dataset_name = dataset_dir.split("/")[-1].replace("/", "")

    yolo_root = "./yolo_temp"

    yolo_seg_path = f"{yolo_root}/{dataset_name}_seg" if args.seg else None
    yolo_bbox_path = f"{yolo_root}/{dataset_name}_bbox" if args.bbox else None

    binmask_to_yolo(f"{dataset_dir}/", args.seg, args.bbox)

if __name__ == '__main__':
    main()
