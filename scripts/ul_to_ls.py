import argparse
from converter.converter import ls_to_ul


def main():
    desc = 'Convert Ultralytics segmentation masks/bounding boxes to Label Studio format'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('label_type', choices=["seg", "bbox"])
    parser.add_argument('ul_dir', type=str, help="Absolute path of the Ultralytics dataset folder")
    parser.add_argument('--ls_base_name', type=str, help="Name of the Label Studio dataset")

    args = parser.parse_args()

    ls_to_ul(
        label_type=args.label_type,
        ls_base_name=args.ls_base_name,
        ul_dir=args.ul_dir,
        reverse=True
    )

if __name__ == '__main__':
    main()
