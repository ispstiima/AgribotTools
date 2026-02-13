"""
Script to convert YOLO format dataset to Ultralytics format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.yolo_to_ul import YoloToUltralytics
from cvtoolkit.formats import TaskType


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO dataset to Ultralytics format with train/val/test splits'
    )
    parser.add_argument('yolo_path', type=str, help='Path to YOLO format dataset')
    parser.add_argument('--ul_path', type=str, default=None, help='Output path for Ultralytics dataset')
    parser.add_argument('--split_ratios', type=float, nargs="*", default=[0.8, 0.2],
                        help='Split ratios (train, val) or (train, val, test)')
    parser.add_argument('--include_test_split', action='store_true', default=False,
                        help='Whether to create a test split')
    parser.add_argument('--image_ext', type=str, default='.jpg,.png',
                        help='Comma-separated list of image extensions')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducible splits')

    args = parser.parse_args()

    yolo_path = Path(args.yolo_path)
    ul_path = Path(args.ul_path) if args.ul_path else yolo_path.parent / f"{yolo_path.name}_ultralytics"

    converter = YoloToUltralytics(
        source_path=yolo_path,
        target_path=ul_path,
        task_type=TaskType.GENERIC
    )

    result = converter.run(
        split_ratios=tuple(args.split_ratios),
        include_test_split=args.include_test_split,
        image_ext=args.image_ext,
        random_seed=args.random_seed,
    )

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()
