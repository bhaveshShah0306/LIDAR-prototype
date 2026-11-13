"""Look at ONE frame through the models' eyes: every raw box, before any of our filters.

Usage:
    uv run python pipeline/debug_frame.py data/rides/2026-11-03_2/video.mp4 1530
    uv run python pipeline/debug_frame.py data/rides/2026-11-03_2/video.mp4 1530 --crop
Saves images to data/debug/ and prints every box.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "data" / "debug"

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("frame", type=int, help="frame number, from review.py's status line")
ap.add_argument("--models", nargs="+", default=["models/fgvd_v1.pt", "yolo11n.pt"])
ap.add_argument("--conf", type=float, default=0.1, help="low on purpose: show weak boxes too")
ap.add_argument("--crop", action="store_true", help="also test a landscape (16:9) crop of the frame")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, frame = cap.read()
if not ok:
    raise SystemExit(f"Could not read frame {args.frame}")

inputs = {"full": frame}
if args.crop:
    h, w = frame.shape[:2]
    band = round(w * 9 / 16)                # height of a 16:9 band as wide as the frame
    y0 = (h - band) // 2                    # take it from the middle
    inputs["crop16x9"] = frame[y0:y0 + band]

OUT.mkdir(parents=True, exist_ok=True)
for model_path in args.models:
    model = YOLO(model_path)
    for label, img in inputs.items():
        r = model.predict(img, conf=args.conf, verbose=False)[0]
        ih, iw = img.shape[:2]
        print(f"\n{Path(model_path).stem} / {label}  ({iw}x{ih})")
        for box, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            x1, y1, x2, y2 = box
            print(f"  {model.names[int(k)]:14s} conf {c:.2f}  "
                  f"w {(x2 - x1) / iw:.0%}  h {(y2 - y1) / ih:.0%}")
        out = OUT / f"f{args.frame}_{Path(model_path).stem}_{label}.jpg"
        cv2.imwrite(str(out), r.plot())     # .plot() draws boxes + names + confidence
        print("  saved", out)