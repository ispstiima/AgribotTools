import argparse


def main():
    desc = 'Converts the segmentation masks in a Label Studio dataset to bounding boxes'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('dataset_path', type=str)

    args = parser.parse_args()

    pass


if __name__ == '__main__':
    main()
