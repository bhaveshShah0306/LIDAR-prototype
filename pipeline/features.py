"""Features: what the alert model is allowed to know about each object at time t.

The golden rule: a feature at time t may only use data from time t and EARLIER.
Using later frames (even by accident, e.g. a centred rolling window) is called LEAKAGE:
the model looks brilliant in testing and useless on the road.

Pipeline position:  events.py -> features.py -> risk.py

Input : data/rides/<ride>/video_events.parquet  (geometry + label), meta.json
Output: data/rides/<ride>/video_features.parquet

Usage:
    uv run python pipeline/features.py data/rides/2026-11-03_2

Done when: every column in FEATURES exists, has no future data, and you can explain each one.

Python you'll practise: groupby + shift/diff/rolling (all backward-looking), categorical columns.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WIDTH = 1280
FPS = 10.0

# The columns risk.py will train on. Add your own ideas; remove ones that don't help.
FEATURES = [
    "ttc_s",      # from geometry.py (box growth)
    "dist_m",     # from geometry.py (ground contact); NaN until calibrated, LightGBM copes with NaN
    "h_frac",     # box height / frame height: bigger = closer
    "area_frac",  # box area / frame area
    "growth",     # change of log(h) over the last 0.5 s
    "drift_x",    # change of centre x over the last 0.5 s, as a fraction of frame width per second
    "aspect",     # box width / height
    "age_s",      # how long this track has existed
    "zone",       # LEFT / CENTRE / RIGHT (categorical)
    "class",      # car / bus_truck / ... / unknown (categorical)
]


def add_features(df, frame_w, frame_h, fps=FPS):
    """Add every FEATURES column that geometry.py didn't already make.

    TODO (always group by track_id and sort by frame first):
      1. h_frac = (y2 - y1) / frame_h;  area_frac = (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
      2. aspect = (x2 - x1) / (y2 - y1)
      3. n = round(0.5 * fps);  log_h = np.log(y2 - y1)
         growth  = log_h - log_h.shift(n)            per track  (groupby(...)["log_h"].shift(n))
         drift_x = (cx - cx.shift(n)) / frame_w / 0.5   per track
      4. age_s = t - first t of the track:  df.groupby("track_id")["t"].transform("min")
      5. zone and class -> pandas categories:  df["zone"] = df["zone"].astype("category")

    Check yourself: .shift(n) looks n rows BACK. What would .shift(-n) do, and why must
    it never appear in this file?
    """
    raise NotImplementedError("TODO: add_features")


def features(ride_dir):
    ride_dir = Path(ride_dir)
    meta = json.loads((ride_dir / "meta.json").read_text())
    w, h = meta["width"], meta["height"]
    frame_w, frame_h = (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)

    df = pd.read_parquet(ride_dir / "video_events.parquet").sort_values(["track_id", "frame"])
    df = add_features(df, frame_w, frame_h)
    df["ride"] = ride_dir.name
    keep = ["ride", "track_id", "frame", "t", *FEATURES, "label"]
    df[keep].to_parquet(ride_dir / "video_features.parquet", index=False)
    print(df[FEATURES].describe(include="all").T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    features(ap.parse_args().ride_dir)
