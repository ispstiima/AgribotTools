from converter.converter import yolo_to_ls

print("=============== yolo_to_ls test ===============")
print("\n\n>>> Converting yolo bbox to ls format...")
yolo_to_ls("../yolo_temp/canon_bbox", "ls_canon_temp_bbox", "bbox")
print("\n\n>>> Converting yolo seg to ls format...")
yolo_to_ls("../yolo_temp/canon_seg", "ls_canon_temp_seg", "seg")
print("\n\n>>> Converting ls bbox to yolo format...")
yolo_to_ls("../yolo_temp/canon_bbox_new", "ls_canon_temp_bbox", "bbox", reverse=True)
print("\n\n>>> Converting ls seg to yolo format...")
yolo_to_ls("../yolo_temp/canon_seg_new", "ls_canon_temp_seg", "seg", reverse=True)
print("\n\n>>> Completed")
