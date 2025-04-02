from converter.converter import yolo_to_ls

print("yolo_to_ls test")
#print("Converting yolo bbox to ls format...")
#yolo_to_ls("/mnt/c/Users/gvero/documents/datasets/canon_yolo_temp/canon_bbox_new", "ls_canon_temp_bbox_new", "bbox")
#print("Converting yolo seg to ls format...")
#yolo_to_ls("/mnt/c/Users/gvero/Documents/Datasets/canon_yolo_temp/canon_seg", "ls_canon_temp_seg", "seg")
#print("Converting ls bbox to yolo format...")
#yolo_to_ls("/mnt/c/Users/gvero/documents/datasets/canon_yolo_temp/canon_bbox_new", "ls_canon_temp_bbox", "bbox", reverse=True)
print("Converting ls seg to yolo format...")
yolo_to_ls("/mnt/c/Users/gvero/Documents/Datasets/canon_yolo_temp/canon_seg_new", "ls_canon_temp_seg", "seg", reverse=True)
print("Completed")
