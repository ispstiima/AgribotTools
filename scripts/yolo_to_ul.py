import argparse
from converter.converter import yolo_to_ul


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation masks/bounding boxes to Ultralytics format')
    parser.add_argument('yolo_path', type=str)
    parser.add_argument('--ul_path', type=str)
    parser.add_argument('--split_ratios', type=float, nargs="*", default=(0.8, 0.2))
    parser.add_argument('--include_test_split', action='store_true', default=False)

    args = parser.parse_args()

    yolo_to_ul(
        yolo_dir=args.yolo_path,
        ul_dir=args.ul_path,
        split_ratios=args.split_ratios,
        include_test_split=args.include_test_split,
    )

if __name__ == '__main__':
    main()
