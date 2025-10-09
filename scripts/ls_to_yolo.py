import argparse
from converter.converter import yolo_to_ls


def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio segmentation masks/bounding boxes to YOLO format')
    parser.add_argument('label_type', choices=["seg", "bbox"])
    parser.add_argument('ls_base_name', type=str)
    parser.add_argument('--yolo_path', type=str)

    args = parser.parse_args()

    yolo_to_ls(
        label_type=args.label_type,
        yolo_dir=args.yolo_path,
        ls_base_name=args.ls_base_name,
        reverse=True
    )

if __name__ == '__main__':
    main()