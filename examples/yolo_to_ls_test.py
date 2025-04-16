from converter.converter import yolo_to_ls

print("=============== yolo_to_ls test ===============")

print("\n\n>>> Converting yolo bbox to ls format...")
ls_bbox_dir = yolo_to_ls("bbox", "../out/canon_yolo_temp/canon_bbox")
print(f"Output path: {ls_bbox_dir}")

print("\n\n>>> Converting yolo seg to ls format...")
ls_seg_dir = yolo_to_ls("seg","../out/canon_yolo_temp/canon_seg")
print(f"Output path: {ls_seg_dir}")

print("\n\n>>> Converting ls bbox to yolo format...")
yolo_bbox_dir = yolo_to_ls("bbox", ls_base_name="canon_bbox_ls", reverse=True)
print(f"Output path: {yolo_bbox_dir}")

print("\n\n>>> Converting ls seg to yolo format...")
yolo_seg_dir = yolo_to_ls("seg", ls_base_name="canon_seg_ls", reverse=True)
print(f"Output path: {yolo_seg_dir}")

print("\n\n>>> Completed")
