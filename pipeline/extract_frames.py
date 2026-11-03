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

import cv2
from ultralytics import YOLO

from detect import CLASS_MAP  # pretrained COCO id -> our class name

LABELS = Path(__file__).resolve().parent.parent / "data" / "labels"
WIDTH = 1280

# YOUR classes for the fine-tuned model. The ORDER matters: the index is the class number in the
# .txt files and in train/data.yaml. Decide once, then never reorder.
CLASSES = ["car", "bus_truck", "two_wheeler", "auto_rickshaw", "person", "animal", "other"]


def sample_frames(video, every_s):
    """Yield (frame_index, resized_frame) every `every_s` seconds.

    TODO:
      1. Open the video, read fps; step = round(every_s * fps).
      2. Read frames in a loop; when frame_index % step == 0, resize like track.py and yield it.
      3. Use `yield`, not a list: a 1-hour ride would not fit in memory as a list of frames.

    Check yourself: what is the difference between a function that `return`s a list
    and one that `yield`s? Which one can handle a 2-hour ride?
    """
    raise NotImplementedError("TODO: sample_frames")


def yolo_lines(result, frame_w, frame_h):
    """Turn one YOLO result into lines of 'class_index cx cy w h' (normalised to 0..1).

    TODO:
      1. For each box: name = CLASS_MAP.get(int(cls)); skip it if None.
      2. class_index = CLASSES.index(name)
      3. cx = (x1 + x2) / 2 / frame_w,  cy = (y1 + y2) / 2 / frame_h,
         w = (x2 - x1) / frame_w,       h = (y2 - y1) / frame_h
      4. Return a list of strings like f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    Check yourself: why normalise to 0..1 instead of keeping pixels?
    """
    raise NotImplementedError("TODO: yolo_lines")


def extract(ride_dirs, every_s, model_name="yolo11n.pt", conf=0.3):
    model = YOLO(model_name)
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
            # TODO: 1. run the model on `frame` (conf=conf, classes=list(CLASS_MAP), verbose=False)
            #       2. save the image: cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
            #       3. write yolo_lines(...) to lbl_dir / f"{name}.txt", one line per box
            #       4. count it: counts[split] = counts.get(split, 0) + 1
            raise NotImplementedError("TODO: save image + labels")
    print(f"Saved frames per split: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dirs", nargs="+")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    args = ap.parse_args()
    extract(args.ride_dirs, args.every)
