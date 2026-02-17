import argparse


def main():
    desc = 'Converts a Label Studio dataset from segmentation masks to bounding boxes'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('ls_base_name', type=str)

    args = parser.parse_args()

    pass


if __name__ == '__main__':
    main()
