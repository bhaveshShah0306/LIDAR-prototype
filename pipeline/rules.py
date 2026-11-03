"""Baseline alert: a hand-written rule. The trained model in risk.py must beat this.

Rule (start here, tune later):
    level 2 (red)   : any object in that zone with TTC < 1.5 s, or closer than 2 m
    level 1 (amber) : any object in that zone with TTC < 2.5 s, or closer than 4 m
    level 0 (off)   : otherwise
CENTRE objects count for BOTH sides (something right behind you can go either way).

Pipeline position:  geometry.py -> rules.py      (risk.py compares itself with this file's output)

Input : data/rides/<ride>/video_geometry.parquet
Output: data/rides/<ride>/video_rules.parquet   frame, t, left_level, right_level
        data/rides/<ride>/video_rules.mp4       (with --video) the ride with two lights drawn on it

Usage:
    uv run python pipeline/rules.py data/rides/2026-11-03_2 --video

Done when: watching video_rules.mp4, the lights come on at moments that look sensible to you.

Python you'll practise: groupby per frame, boolean masks, a small state machine (hold time).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RED_TTC, AMBER_TTC = 1.5, 2.5   # seconds
RED_DIST, AMBER_DIST = 2.0, 4.0  # metres
HOLD_S = 0.5                     # keep a light on at least this long, so it doesn't flicker


def box_level(ttc_s, dist_m):
    """Level 0/1/2 for each box (Series in, Series out).

    TODO:
      1. red = (ttc_s < RED_TTC) | (dist_m < RED_DIST)        # NaN compares as False, which is what we want
      2. amber = (ttc_s < AMBER_TTC) | (dist_m < AMBER_DIST)
      3. return np.select([red, amber], [2, 1], default=0) as a Series with ttc_s's index
    """
    raise NotImplementedError("TODO: box_level")


def frame_levels(df):
    """One row per frame: the highest level on the rider's left and right.

    TODO:
      1. df["level"] = box_level(df["ttc_s"], df["dist_m"])
      2. left  = level where zone is "LEFT" or "CENTRE", else 0;  right = same for "RIGHT"/"CENTRE"
      3. Group by ["frame", "t"] and take the max of left and right.
      4. Frames with no boxes at all are missing: reindex so every analysed frame has a row (level 0).
         (Hint: the frames list can come from the unique frames of the input, which already
          has every frame that had at least one box; that's good enough to start.)
      5. Return a DataFrame with columns frame, t, left_level, right_level.
    """
    raise NotImplementedError("TODO: frame_levels")


def apply_hold(levels, fps=10.0, hold_s=HOLD_S):
    """Keep each light on for at least hold_s after it was last triggered (removes flicker).

    TODO:
      1. hold = round(hold_s * fps) frames.
      2. For each column (left_level, right_level): a rolling MAX over the last `hold` rows
         does exactly this:  levels[col].rolling(hold, min_periods=1).max().astype(int)

    Check yourself: why does a warning light need hold time, while the model's raw output doesn't?
    """
    raise NotImplementedError("TODO: apply_hold")


def draw_lights(frame, left, right):
    """Draw two circles at the top of the frame: off = grey, 1 = amber, 2 = red.

    TODO: colours are BGR in OpenCV: grey (80, 80, 80), amber (0, 191, 255), red (0, 0, 255).
          cv2.circle(frame, centre, radius, colour, thickness=-1)   # -1 fills it
          Left light on the left of the frame, right light on the right.
    """
    raise NotImplementedError("TODO: draw_lights")


def render(ride_dir, levels, width=1280):
    """Write video_rules.mp4: every analysed frame with the two lights drawn.

    TODO (reuse the reading loop from track.py):
      1. Open video.mp4; step = round(src_fps / 10); resize frames like track.py.
      2. Build a dict {frame: (left, right)} from `levels` for fast lookups.
      3. For each analysed frame: draw_lights(...), write it with cv2.VideoWriter.
    """
    raise NotImplementedError("TODO: render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    ap.add_argument("--video", action="store_true", help="also write video_rules.mp4")
    args = ap.parse_args()

    ride = Path(args.ride_dir)
    geo = pd.read_parquet(ride / "video_geometry.parquet")
    levels = apply_hold(frame_levels(geo))
    levels.to_parquet(ride / "video_rules.parquet", index=False)
    on = (levels[["left_level", "right_level"]] > 0).mean()
    print(f"Light on: left {on['left_level']:.0%} of frames, right {on['right_level']:.0%} of frames")
    if args.video:
        render(ride, levels)
