import argparse
from converter.converter import yolo_to_ul


def main():
    parser = argparse.ArgumentParser(description='Convert Ultralytics dataset to YOLO format')
    parser.add_argument('ul_path', type=str)
    parser.add_argument('--yolo_path', type=str)

    args = parser.parse_args()

    yolo_to_ul(
        yolo_dir=args.yolo_path,
        ul_dir=args.ul_path,
        reverse=True
    )

if __name__ == '__main__':
    main()
