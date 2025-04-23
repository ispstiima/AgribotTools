from converter.converter import yolo_to_ul

print("=============== yolo_to_ul test ===============")
print("\n\n>>> Converting yolo to ul format...")
yolo_to_ul("../out/canon_yolo_temp/canon_bbox", "../out/canon_ul_temp/canon_bbox", random_seed=42)
print("\n\n>>> Converting ul to yolo format...")
yolo_to_ul("../out/canon_yolo_temp/canon_bbox_new", "../out/canon_ul_temp/canon_bbox", random_seed=42, reverse=True)
