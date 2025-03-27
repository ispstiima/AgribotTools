# AgribotTools

## Converter

### Acronyms and Definitions

| Acronym | Explaination |
| ------- | ------------ |
| LS | **Label Studio**: the software used for labeling. |
| UL | **Ultralytics**: target library for object detection and segmentation. |
| Binmask | **Binary mask**: binary image where a $1$ represents a foreground pixel, while a $0$ represents a background pixel. |
| Segmask | **Segmentation mask**: mask used to highlight an object of interest. |
| Bbox | **Bounding box**: box used to highlight an object of interest. |

### Format Definition

##### BinMask format

It is a folder containing:

* A subfolder `images` containing the labeled images in `jpg` or `png` format.
* A subfolder `labels` containing, for each image in the `images` subfolder, a `png` image, a file with the same name of the corresponding image, which describes the segmask associated to the corresponding image.

##### YOLO format

It is a folder containing:

* A subfolder `images` containing the labeled images in `jpg` or `png` format.
* A subfolder `labels` containing, for each image in the `images` subfolder, a text file with the same name of the corresponding image, in which each row describes a segmmask or a bbox in the following format:
   ```
   <class_id><x1><y1><x2><y2>...<xn><yn>
   <class_id><x_center><y_canter><width><height>
   ```
* A text file `classes.txt` highlighting the labelled classes, in which each row contains a single string with the name of the corresponding class. The index of each row represents the identifier of the class contained in the `class_id` field of the text file contained in `labels`.

##### LS format

It is a folder containing:

* A subfolder `images` containing the labeled images in `jpg` or `png` format.
* A file in `json` format, containing information relative to images and their respective labels.
* A file in `xml` format for the configuration of the labeling interface:
   ```xml
   <View>
      <!-- View the image to be labeled -->
      <Image name="image" value="$image" />
      <!-- Define the bbox's label -->
      <Labels name="label" toName="image">
         <Label value="Object" />
      </Labels>
      
      <!-- Tool for drawing bboxes -->
      <RectangleLabels name="bbox" toName="image">
         <Label value="Object" />
      </RectangleLabels>
   </View>
   ```

##### UL format

It is a folder named as the dataset, e.g., `xylella`, containing:

* A subfolder `train` containing the portion of the dataset where training data will be stored, containing:
   * A subfolder `images` containing the images in `jpg` or `png` format.
   * A subfolder `labels` containing, for each image in `images`, the corresponding labels in YOLO format.
* A subfolder `val` containing the portion of the dataset where training data will be stored, containing:
   * A subfolder `images` containing the images in `jpg` or `png` format.
   * A subfolder `labels` containing, for each image in `images`, the corresponding labels in YOLO format.
* A configuration file in `yaml` format with the same name of the dataset formatted as follows:
   ```yaml
   # Dataset name
   path: /path/to/dataset        # Path of the dataset
   train: /train/images          # Path of training images (relative to path)
   val: /val/images              # Path of validation images  (relative to path)
   test: /test/images            # Path of test images (relative to path, optional)

   # Classes names
   names:
      0: first class
      1: second class
      ...
   ```

> **Note**: the `train` and `val` subfolders share the same structure.

### Use cases

##### Edit segmentations masks on Label Studio

1. Import the dataset on Label Studio using the module `binmask_to_ls`.
2. Edit the segmentation masks.

##### Edit bounding boxes on Label Studio

1. Import the dataset on Label Studio:
   a. Using the `binmask_to_ls` module, if the labels are in the binmask format.
   b. Using the moduel `seg_ls_to_bbox_ls`, if the labels are segmasks in LS format.
   c. Directly from the dataset saved in the root folder.
2. Edit bboxes.

##### Convert LS labels in an UL dataset

1. Select JSON as export type.
2. Convert from the LS format to the UL format using the `ls_to_ul` module.

## References

* [Ultralytics - Object Detection Datasets Overview](https://docs.ultralytics.com/datasets/detect/)
* [Ultralytics - Instance Segmentation Datasets Overview](https://docs.ultralytics.com/datasets/segment/)
* [Label Studio - Understanding the Label Studio JSON format](https://labelstud.io/blog/understanding-the-label-studio-json-format/#breaking-down-the-label-studio-json-format)
* [Label Studio - Labeling configuration](https://labelstud.io/templates/named_entity#Labeling-Configuration)