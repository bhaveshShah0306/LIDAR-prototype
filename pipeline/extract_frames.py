"""Pick frames to label and pre-label them with the current model (Saturday's dataset task).

Pre-labelling means the pretrained YOLO draws boxes first; in Label Studio you only fix
mistakes and add what it can't know (auto-rickshaws, `other`). Much faster than drawing everything.

Input : data/rides/<ride>/video.mp4 (+ meta.json "split": "train" or "test")
Output: data/labels/<split>/images/<ride>_f<frame>.jpg
        data/labels/<split>/labels/<ride>_f<frame>.txt   YOLO format: class cx cy w h (all 0..1)
        data/labels/classes.txt

Usage:
    uv run python pipeline/extract_frames.py data/rides/2026-10-28_1 data/rides/2026-11-03_2 --every 2

Done when: ~250 train + ~50 test images exist, each with a .txt next to it, and they import
into Label Studio with boxes already drawn.

Python you'll practise: sampling, list/dict building, writing a simple file format by hand.
"""
import argparse
import json
from pathlib import Path
from xml.parsers.expat import model

import cv2
from ultralytics import YOLO

from detect import class_map_for # pretrained COCO id -> our class name

LABELS = Path(__file__).resolve().parent.parent / "data" / "labels"
WIDTH = 1280

# YOUR classes for the fine-tuned model. The ORDER matters: the index is the class number in the
# .txt files and in train/data.yaml. Decide once, then never reorder.
CLASSES = ["car", "bus_truck", "two_wheeler", "auto_rickshaw", "person", "animal", "other"]


def sample_frames(video, every_s, width=WIDTH):
    """Yield (frame_index, resized_frame) every `every_s` seconds."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(every_s * fps))                 # frames between two samples
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if frame.shape[1] > width:        # same resize as track.py
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            yield idx, frame                         # hand back the index and the frame
        idx += 1
    cap.release()


def yolo_lines(result, class_map):
    """YOLO label lines for one result, using OUR class numbers."""
    lines = []
    for (cx, cy, w, h), k in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist()):
        name = class_map.get(int(k))
        if name is None:
            continue
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")   # our class number for `name`
    return lines


def extract(ride_dirs, every_s, model_name="yolo11n.pt", conf=0.3):
    model = YOLO(model_name)
    class_map = class_map_for(model) 
    LABELS.mkdir(parents=True, exist_ok=True)
    (LABELS / "classes.txt").write_text("\n".join(CLASSES) + "\n") 
    counts = {}
    for ride in map(Path, ride_dirs):
        meta = json.loads((ride / "meta.json").read_text())
        split = meta.get("split") or "train"
        img_dir, lbl_dir = LABELS / split / "images", LABELS / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in sample_frames(ride / "video.mp4", every_s):
            name = f"{ride.name}_f{idx:06d}"
            r = model.predict(frame, conf=conf, verbose=False)[0]
            cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
            (lbl_dir / f"{name}.txt").write_text("\n".join(yolo_lines(r, class_map)) + "\n")
            counts[split] = counts.get(split, 0) + 1
    print(f"Saved frames per split: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dirs", nargs="+")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    args = ap.parse_args()
    extract(args.ride_dirs, args.every)
