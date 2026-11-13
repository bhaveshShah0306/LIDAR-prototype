"""Save one crop per track of a class, to check by eye what the model calls that class.

Usage:
    uv run python pipeline/track_crops.py data/rides/2026-11-12_1 bus_truck
Saves data/debug/crops/<ride>/<class>/track<ID>_<t>s_conf<c>.jpg
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 1280   # same resize as track.py
PAD = 0.1      # show 10% extra around the box, for context

ap = argparse.ArgumentParser()
ap.add_argument("ride_dir")
ap.add_argument("cls")
args = ap.parse_args()
ride = Path(args.ride_dir)

df = pd.read_parquet(ride / "video_tracks.parquet")
df = df[df["class"] == args.cls].sort_values(["track_id", "frame"])
# pick the MIDDLE row of each track: its most typical view
df["pos"] = df.groupby("track_id").cumcount()
df["size"] = df.groupby("track_id")["frame"].transform("size")
mid = df[(df["pos"] == df["size"] // 2) & (df["size"] >= 5)]      # skip flickers

out = ROOT / "data" / "debug" / "crops" / ride.name / args.cls
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(ride / "video.mp4"))
for r in mid.itertuples():
    cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
    ok, frame = cap.read()
    if not ok:
        continue
    if frame.shape[1] > WIDTH:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    H, W = frame.shape[:2]
    pw, ph = PAD * (r.x2 - r.x1), PAD * (r.y2 - r.y1)
    x1, y1 = max(0, int(r.x1 - pw)), max(0, int(r.y1 - ph))
    x2, y2 = min(W, int(r.x2 + pw)), min(H, int(r.y2 + ph))
    name = f"{r.size:03d}f_track{r.track_id}_{r.t:.0f}s_conf{r.conf:.2f}.jpg"
    cv2.imwrite(str(out / name), frame[y1:y2, x1:x2])

print(f"Saved {len(mid)} crops to {out}")