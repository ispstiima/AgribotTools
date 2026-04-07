"""
Script to convert YOLO format to Label Studio format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.yolo_to_ls import YoloToLabelStudio
from cvtoolkit.formats import TaskType
from utils import TASK_TYPE_MAP


def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO segmentation masks/bounding boxes to Label Studio format'
    )
    parser.add_argument('task_type', choices=["seg", "bbox"],
                        help='Task type: "seg" for segmentation, "bbox" for detection')
    parser.add_argument('yolo_path', type=str,
                        help='Path to the YOLO dataset directory')
    parser.add_argument('--ls_path', type=str, default=None,
                        help='Output path for Label Studio dataset')
    parser.add_argument('--image_root_url', type=str, default=None,
                        help='Root URL path where images will be hosted in Label Studio')
    args = parser.parse_args()

    source = Path(args.yolo_path)
    target = Path(args.ls_path) if args.ls_path else source.parent / f"{source.name}_ls"
    task_type = TASK_TYPE_MAP[args.task_type]

    converter = YoloToLabelStudio(source, target, task_type)
    result = converter.run()

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()