import argparse
from tools.converter import convert_binary_masks_to_yolo


def main():
    parser = argparse.ArgumentParser(description='Convert binary segmentation maps to masks and bounding boxes in yolo format')
    parser.add_argument('mask_path', type=str)
    parser.add_argument('--output_seg_path', default="./out/seg", type=str)
    parser.add_argument('--output_box_path', default="./out/box", type=str)

    args = parser.parse_args()

    convert_binary_masks_to_yolo(args.mask_path, args.output_seg_path, args.output_box_path)


if __name__ == '__main__':
    main()
