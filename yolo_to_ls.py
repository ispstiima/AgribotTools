import argparse
from label_studio_sdk.converter.imports import yolo


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO format to Label Studio format")
    parser.add_argument("--dataset_path", type=str, help="YOLO Dataset path")
    args = parser.parse_args()

    dataset_dir = args.dataset_path
    dataset_name = dataset_dir.split("/")[-1].replace("/", "")

    image_root = f"/data/local-files/?d={dataset_name}/images"
    output_file = f"{dataset_dir}/task.json"

    yolo.convert_yolo_to_ls(dataset_dir, output_file, image_root_url=image_root)


if __name__ == "__main__":
    main()
