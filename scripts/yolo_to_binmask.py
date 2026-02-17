import argparse


def main():
    # TODO implementare yolo_to_binmask
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation masks to binary mask images')
    parser.add_argument('yolo_path', type=str, help='Path to the dataset directory containing the YOLO dataset')
    args = parser.parse_args()

if __name__ == '__main__':
    main()
