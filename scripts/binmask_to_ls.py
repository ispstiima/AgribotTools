"""
Script to convert binary segmentation masks to Label Studio format.
"""
import argparse
from pathlib import Path
from cvtoolkit.conversions.binmask_to_ls import BinmaskToLabelStudio
from utils import TASK_TYPE_MAP


def main():
    parser = argparse.ArgumentParser(
        description='Convert binary segmentation masks to Label Studio format'
    )
    parser.add_argument('binmask_path', type=str,
                        help='Path to the dataset directory containing images and masks')
    parser.add_argument('task_type', choices=["seg", "bbox"],
                        help='Task type: "seg" for segmentation, "bbox" for detection')
    parser.add_argument('--ls_path', type=str, default=None,
                        help='Output path for Label Studio dataset')
    args = parser.parse_args()

    source = Path(args.binmask_path)
    target = Path(args.ls_path) if args.ls_path else source.parent / f"{source.name}_ls"
    task_type = TASK_TYPE_MAP[args.task_type]

    converter = BinmaskToLabelStudio(source, target, task_type)
    result = converter.run()

    print(f"Conversion complete: {result}")


if __name__ == '__main__':
    main()
