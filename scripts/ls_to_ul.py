import argparse
from cvtoolkit.converter import ls_to_ul


def main():
    desc = 'Convert Label Studio segmentation masks/bounding boxes to Ultralytics format'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('label_type', choices=["seg", "bbox"])
    parser.add_argument('ls_base_name', type=str, help="Name of the Label Studio dataset")
    parser.add_argument('--ul_dir', type=str, help="Absolute path of the Ultralytics dataset folder")
    parser.add_argument('--split_ratios', type=float, nargs="*", default=(0.8, 0.2))
    parser.add_argument('--include_test_split', action='store_true', default=False)

    args = parser.parse_args()

    ls_to_ul(
        label_type=args.label_type,
        ls_base_name=args.ls_base_name,
        ul_dir=args.ul_dir,
        split_ratios=args.split_ratios,
        include_test_split=args.include_test_split,
    )

if __name__ == '__main__':
    main()
