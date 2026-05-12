"""
Script to convert Label Studio format to YOLO format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.ls_to_yolo import LabelStudioToYolo
from cvtoolkit.formats import TaskType
from utils import TASK_TYPE_MAP


def main():
    parser = argparse.ArgumentParser(
        description='Convert Label Studio segmentation masks/bounding boxes to YOLO format'
    )
    parser.add_argument('task_type', choices=["seg", "bbox"],
                        help='Task type: "seg" for segmentation, "bbox" for detection')
    parser.add_argument('ls_path', type=str,
                        help='Path to the Label Studio dataset directory')
    parser.add_argument('--yolo_path', type=str, default=None,
                        help='Output path for YOLO dataset')
    args = parser.parse_args()

    source = Path(args.ls_path)
    target = Path(args.yolo_path) if args.yolo_path else source.parent / f"{source.name}_yolo"
    task_type = TASK_TYPE_MAP[args.task_type]

    converter = LabelStudioToYolo(source, target, task_type)
    result = converter.run()

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()