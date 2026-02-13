"""
Script to convert Label Studio format to Ultralytics format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.ls_to_ul import LabelStudioToUltralytics
from cvtoolkit.formats import TaskType


TASK_TYPE_MAP = {
    "seg": TaskType.SEGMENTATION,
    "bbox": TaskType.DETECTION,
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio segmentation masks/bounding boxes to Ultralytics format'
    )
    parser.add_argument('task_type', choices=["seg", "bbox"],
                        help='Task type: "seg" for segmentation, "bbox" for detection')
    parser.add_argument('ls_path', type=str,
                        help='Path to the Label Studio dataset directory')
    parser.add_argument('--ul_path', type=str, default=None,
                        help='Output path for Ultralytics dataset')
    parser.add_argument('--split_ratios', type=float, nargs="*", default=[0.8, 0.2],
                        help='Split ratios (train, val) or (train, val, test)')
    parser.add_argument('--include_test_split', action='store_true', default=False,
                        help='Whether to create a test split')
    args = parser.parse_args()

    source = Path(args.ls_path)
    target = Path(args.ul_path) if args.ul_path else source.parent / f"{source.name}_ultralytics"
    task_type = TASK_TYPE_MAP[args.task_type]

    converter = LabelStudioToUltralytics(source, target, task_type)
    result = converter.run(
        split_ratios=tuple(args.split_ratios),
        include_test_split=args.include_test_split,
    )

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()
