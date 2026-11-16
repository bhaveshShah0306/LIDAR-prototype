"""Convert our YOLO labels into X-AnyLabeling / LabelMe JSON files next to the images.

Usage:
    uv run python pipeline/yolo_to_labelme.py data/labels/own/train
"""
import json
import sys
from pathlib import Path

import cv2

split = Path(sys.argv[1])
classes = (split.parent / "classes.txt").read_text().split()
written = 0
for img_path in sorted((split / "images").glob("*.jpg")):
    h, w = cv2.imread(str(img_path)).shape[:2]
    shapes = []
    label_file = split / "labels" / (img_path.stem + ".txt")
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            k, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            shapes.append({
                "label": classes[int(k)],
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],   # 4 corners of the rectangle
                "group_id": None, "description": "", "difficult": False,
                "shape_type": "rectangle", "flags": {},
            })
    doc = {"version": "3.3.10", "flags": {}, "shapes": shapes, "imagePath": img_path.name,
           "imageData": None, "imageHeight": h, "imageWidth": w}
    img_path.with_suffix(".json").write_text(json.dumps(doc, indent=2))
    written += 1

print(f"Wrote {written} JSON files next to the images in {split / 'images'}")