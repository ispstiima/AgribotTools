from converter.converter import ls_to_ul

print("=============== ls_to_ul test ===============")

print("\n\n>>> Converting ls bbox to ul format...")
ul_bbox_dir = ls_to_ul("bbox", ls_base_name="canon_bbox",)
print(f"Output path: {ul_bbox_dir}")

print("\n\n>>> Converting ls seg to ul format...")
ul_seg_dir = ls_to_ul("seg", ls_base_name="canon_seg",)
print(f"Output path: {ul_seg_dir}")

print("\n\n>>> Converting ul bbox to ls format...")
ls_bbox_dir = ls_to_ul("bbox", ul_dir=ul_bbox_dir, reverse=True)
print(f"Output path: {ls_bbox_dir}")

print("\n\n>>> Converting ul seg to ls format...")
ls_seg_dir = ls_to_ul("seg", ul_dir=ul_seg_dir, reverse=True)
print(f"Output path: {ls_seg_dir}")

print("\n\n>>> Completed")