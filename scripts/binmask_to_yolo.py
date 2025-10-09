import argparse
from converter.converter import binmask_to_yolo


def main():
    parser = argparse.ArgumentParser(description='Convert binary segmentation masks to YOLO format')
    parser.add_argument('binmask_path', type=str, help='Path to the dataset directory containing the images and masks')
    parser.add_argument('--seg',  '-s', action='store_true')
    parser.add_argument('--bbox', '-b', action='store_true')
    parser.add_argument('--yolo_path', type=str)
    args = parser.parse_args()
    binmask_to_yolo(args.binmask_path, args.seg, args.bbox, args.yolo_path)

if __name__ == '__main__':
    main()
