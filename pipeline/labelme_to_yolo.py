"""Convert X-AnyLabeling / LabelMe JSON files (next to the images) into YOLO .txt labels.

Usage:
    uv run python pipeline/labelme_to_yolo.py data/labels/own/train
Overwrites data/labels/own/train/labels/*.txt with the corrected labels from the .json files.
"""
import json
import sys
from pathlib import Path

split = Path(sys.argv[1])
classes = (split.parent / "classes.txt").read_text().split()
out_dir = split / "labels"
out_dir.mkdir(exist_ok=True)

written, unknown = 0, {}
for js in sorted((split / "images").glob("*.json")):
    doc = json.loads(js.read_text())
    w, h = doc["imageWidth"], doc["imageHeight"]
    lines = []
    for shape in doc["shapes"]:
        if shape.get("shape_type") != "rectangle":
            continue
        name = shape["label"]
        if name not in classes:                      # e.g. the junk labels from the bad import
            unknown[name] = unknown.get(name, 0) + 1
            continue
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        x1, x2 = max(0, min(xs)), min(w, max(xs))    # clip to the image
        y1, y2 = max(0, min(ys)), min(h, max(ys))
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{classes.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (out_dir / (js.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    written += 1

print(f"Wrote {written} label files to {out_dir}")
if unknown:
    print("WARNING, boxes with unknown labels were skipped:", unknown)