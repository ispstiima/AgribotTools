import argparse
from converter import binmask_to_yolo


def main():
    parser = argparse.ArgumentParser(description='Convert binary segmentation masks to Label Studio format')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory containing the images and masks')
    parser.add_argument('--seg',  '-s', action='store_true')
    parser.add_argument('--bbox', '-b', action='store_true')
    args = parser.parse_args()
    binmask_to_yolo(args.dataset_path, args.seg , args.bbox)

if __name__ == '__main__':
    main()
