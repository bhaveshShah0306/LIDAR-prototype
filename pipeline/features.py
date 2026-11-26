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
    "ttc_s", "ttc_1s",           # box growth over 0.5 s and 1 s
    "dist_m",                    # NaN until calibrated (LightGBM copes with NaN)
    "lat_m", "gap_m", "in_corridor",   # add_lateral; NaN/False until calibrated
    "h_frac", "area_frac", "aspect",   # size and shape
    "growth", "drift_x",         # rates per second, past-only
    "age_s", "conf",             # track age, detector confidence
    "zone", "class",             # categorical
]



def add_features(df, frame_w, frame_h, fps=FPS, max_ttc=20.0):
    """Every FEATURES column geometry.py didn't make. PAST ONLY: shift(n) looks n rows back."""
    df = df.sort_values(["track_id", "frame"]).copy()
    by = df["track_id"]
    n = max(1, round(0.5 * fps))                                  # 5 rows = 0.5 s
    box_w, box_h = df["x2"] - df["x1"], df["y2"] - df["y1"]

    df["h_frac"] = box_h / frame_h
    df["area_frac"] = box_w * box_h / (frame_w * frame_h)
    df["aspect"] = box_w / box_h

    log_h = np.log(box_h)
    dt = df["t"] - df["t"].groupby(by).shift(n)                   # real seconds back
    df["growth"] = (log_h - log_h.groupby(by).shift(n)) / dt      # 1/s
    df["drift_x"] = (df["cx"] - df["cx"].groupby(by).shift(n)) / frame_w / dt   # frame widths/s

    dt2 = df["t"] - df["t"].groupby(by).shift(2 * n)              # 1 s window
    rate1 = (log_h - log_h.groupby(by).shift(2 * n)) / dt2
    df["ttc_1s"] = np.where(rate1 > 0, 1 / rate1, np.inf)
    df.loc[df["ttc_1s"] > max_ttc, "ttc_1s"] = np.inf

    df["age_s"] = df["t"] - df.groupby("track_id")["t"].transform("min")   # first t of the track
    for col in ["zone", "class"]:
        df[col] = df[col].astype("category")
    return df


def features(ride_dir):
    ride_dir = Path(ride_dir)
    meta = json.loads((ride_dir / "meta.json").read_text())
    w, h = meta["width"], meta["height"]
    frame_w, frame_h = (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)

    src = ride_dir / "video_events.parquet"
    if not src.exists():                      # no label yet: events.py needs calibration
        src = ride_dir / "video_geometry.parquet"
    df = pd.read_parquet(src)
    df = add_features(df, frame_w, frame_h)
    df["ride"] = ride_dir.name
    keep = ["ride", "track_id", "frame", "t", *FEATURES] + (["label"] if "label" in df else [])
    df[keep].to_parquet(ride_dir / "video_features.parquet", index=False)
    print(df[FEATURES].describe(include="all").T.to_string())

    # def features(ride_dir):
    # ride_dir = Path(ride_dir)
    # meta = json.loads((ride_dir / "meta.json").read_text())
    # w, h = meta["width"], meta["height"]
    # frame_w, frame_h = (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)

    # src = ride_dir / "video_events.parquet"
    # if not src.exists():                      # no label yet: events.py needs calibration
    #     src = ride_dir / "video_geometry.parquet"
    # df = pd.read_parquet(src)
    # df = add_features(df, frame_w, frame_h)   # add_features sorts by track and frame itself
    # df["ride"] = ride_dir.name

    # keep = ["ride", "track_id", "frame", "t", *FEATURES] + (["label"] if "label" in df else [])
    # df[keep].to_parquet(ride_dir / "video_features.parquet", index=False)
    # print(df[FEATURES].describe(include="all").T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    features(ap.parse_args().ride_dir)
