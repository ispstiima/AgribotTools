from label_studio_sdk.converter.imports import yolo

dataset_name = "Xylella"

dataset_dir = f"/mnt/d/Dataset/LabelStudio/{dataset_name}"
image_root = f"/data/local-files/?d={dataset_name}/images"
output_file = f"{dataset_dir}/task.json"

yolo.convert_yolo_to_ls(dataset_dir, output_file, image_root_url=image_root)
