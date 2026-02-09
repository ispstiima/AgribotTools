"""
Script to convert Ultralytics format dataset back to standard YOLO format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.ul_to_yolo import UltralyticsToYolo
from cvtoolkit.formats import TaskType


def main():
    parser = argparse.ArgumentParser(
        description='Convert Ultralytics dataset back to standard YOLO format'
    )
    parser.add_argument('ul_path', type=str, help='Path to Ultralytics format dataset')
    parser.add_argument('--yolo_path', type=str, default=None, help='Output path for YOLO dataset')

    args = parser.parse_args()

    ul_path = Path(args.ul_path)
    yolo_path = Path(args.yolo_path) if args.yolo_path else ul_path.parent / f"{ul_path.name}_yolo"

    converter = UltralyticsToYolo(
        source_path=ul_path,
        target_path=yolo_path,
        task_type=TaskType.GENERIC
    )
    
    result = converter.run()
    
    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()
