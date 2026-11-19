"""Check a YOLO dataset split before training.

Usage:
    uv run python pipeline/check_dataset.py data/labels/own/train data/labels/own/classes.txt
"""
import sys
from collections import Counter
from pathlib import Path

split, classes_file = Path(sys.argv[1]), Path(sys.argv[2])
classes = classes_file.read_text().split()
images = {p.stem for p in (split / "images").glob("*.jpg")}
labels = {p.stem for p in (split / "labels").glob("*.txt")}

print(f"classes ({len(classes)}): {classes}")
print(f"images: {len(images)}   label files: {len(labels)}")
print(f"images without labels: {len(images - labels)}   labels without images: {len(labels - images)}")

counts, problems, empty = Counter(), [], 0
for stem in sorted(labels & images):
    lines = [l for l in (split / "labels" / f"{stem}.txt").read_text().splitlines() if l.strip()]
    if not lines:
        empty += 1
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{stem}.txt line {n}: expected 5 values, got {len(parts)}")
            continue
        k, *nums = parts
        if not k.isdigit() or int(k) >= len(classes):
            problems.append(f"{stem}.txt line {n}: invalid class {k}")
            continue
        cx, cy, w, h = map(float, nums)
        if not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            problems.append(f"{stem}.txt line {n}: values outside 0..1 or empty box")
        counts[classes[int(k)]] += 1

print(f"images with no boxes (background): {empty}")
print("boxes per class:", dict(counts))
print(f"problems: {len(problems)}")
for p in problems[:10]:
    print("  ", p)