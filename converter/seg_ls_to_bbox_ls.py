import argparse
import logging
from converter.converter import ls_to_yolo, seg_yolo_to_bbox_yolo, yolo_to_ls


log = logging.getLogger("SegLS-To-BBoxLS")


def main():
    desc = 'Converts the segmentation masks in a Label Studio dataset to bounding boxes'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('dataset_name', type=str)

    args = parser.parse_args()

    log.info("Starting conversion...")

    ls_to_yolo(ls_data_name=args.dataset_name, "./yolo_root/Test_YOLOseg", "seg")
    seg_yolo_to_bbox_yolo("./yolo_root", "Test_YOLOseg")
    yolo_to_ls("./yolo_root/Test_YOLOseg", "Test_LSbbox", "bbox")

    log.info("Execution completed.")


if __name__ == '__main__':
    main()
