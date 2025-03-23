# AgribotTools

## Converter

### Acronyms and Definitions

* Label Studio: LS
* Ultralytics: UL
* Segmentation masks: segmasks
* Binary masks: binmasks
* Bounding boxes: bboxes

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
