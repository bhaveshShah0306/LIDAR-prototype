"""Draw YOLO labels back onto their images, to check a dataset by eye before training.

Usage:
    uv run python pipeline/check_labels.py data/labels/fgvd/train --n 3

Saves data/labels/check_<image name>.jpg for each picked image.
Every box should sit exactly on its vehicle or person. If boxes are shifted, too big or
in the wrong place, the converter has a bug: fix it BEFORE training.
"""
import argparse
import random
from pathlib import Path

import cv2

from extract_frames import CLASSES

OUT = Path(__file__).resolve().parent.parent / "data" / "labels"
COLOURS = [(0, 220, 0), (200, 0, 200), (255, 160, 0), (0, 255, 255),
           (0, 200, 255), (0, 120, 255), (180, 180, 180)]  # one BGR colour per class index


def yolo_to_pixels(cx, cy, w, h, img_w, img_h):
    """Turn one YOLO box (centre + size, 0..1) back into pixel corners (x1, y1, x2, y2).

    This is the REVERSE of what vehicle_lines() does.
    """
    x1 = int((cx - w / 2) * img_w)      # left edge:   centre minus half the width, then times the image width
    y1 = int((cy - h / 2) * img_h)      # top edge:    centre minus half the height, then times the image height
    x2 = int((cx + w / 2) * img_w)      # right edge:  centre plus half the width, times the image width
    y2 = int((cy + h / 2) * img_h)      # bottom edge: centre plus half the height, times the image height
    return x1, y1, x2, y2   # OpenCV needs whole pixel numbers


def draw(img_path, label_path):
    """Return the image with every box from the label file drawn on it."""
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]       # height and width from the image array (hint: img.shape)

    for line in label_path.read_text().splitlines():
        if not line.strip():            # skip empty lines (background images have none)
            continue
        parts = line.split()            # "4 0.773737 0.645069 0.098576 0.370852" -> 5 strings
        cls = int(parts[0])              # first part, as an int
        cx, cy, w, h = map(float, parts[1:5])  # the other four parts, as floats (hint: map(float, ...))

        x1, y1, x2, y2 = yolo_to_pixels(cx, cy, w, h, img_w, img_h)
        print(line, "->", x1, y1, x2, y2, "image", img_w, img_h)
        colour = COLOURS[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, CLASSES[cls], (x1, max(15, y1 - 5)),     # the class NAME (hint: CLASSES)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", help="e.g. data/labels/fgvd/train (contains images/ and labels/)")
    ap.add_argument("--n", type=int, default=3, help="how many random images to check")
    args = ap.parse_args()

    split = Path(args.split_dir)
    images = sorted((split / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No .jpg files in {split / 'images'}. Pass the split folder, e.g. data/labels/fgvd/train")
    for img_path in random.sample(images, args.n):
        label_path = split / "labels" / (img_path.stem + ".txt")  # same name as the image, but in labels/ and ending .txt
        out = OUT / f"check_{img_path.name}"
        cv2.imwrite(str(out), draw(img_path, label_path))
        print("Saved", out)