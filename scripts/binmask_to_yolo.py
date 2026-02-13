"""
Script to convert binary segmentation masks to YOLO format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.binmask_to_yolo import BinmaskToYolo
from cvtoolkit.formats import TaskType


TASK_TYPE_MAP = {
    "seg": TaskType.SEGMENTATION,
    "bbox": TaskType.DETECTION,
}


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary segmentation masks to YOLO format'
    )
    parser.add_argument('binmask_path', type=str,
                        help='Path to the dataset directory containing images and masks')
    parser.add_argument('task_type', choices=["seg", "bbox"],
                        help='Task type: "seg" for segmentation, "bbox" for detection')
    parser.add_argument('--yolo_path', type=str, default=None,
                        help='Output path for YOLO dataset')
    args = parser.parse_args()

    source = Path(args.binmask_path)
    target = Path(args.yolo_path) if args.yolo_path else source.parent / f"{source.name}_yolo"
    task_type = TASK_TYPE_MAP[args.task_type]

    converter = BinmaskToYolo(source, target, task_type)
    result = converter.run()

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()
