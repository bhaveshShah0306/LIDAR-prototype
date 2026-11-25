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
from detect import COLOURS, draw
from geometry import SIDE_FRACTION
RED_TTC, AMBER_TTC = 1.5, 2.5   # seconds
RED_DIST, AMBER_DIST = 2.0, 4.0  # metres
HOLD_S = 0.5                     # keep a light on at least this long, so it doesn't flicker
GREY, AMBER, RED = (80, 80, 80), (0, 191, 255), (0, 0, 255)   # BGR: OpenCV's colour order


def box_level(ttc_s, dist_m):
    """Level per box: 2 red, 1 amber, 0 off (Series in, Series out).

    TTC alone can trigger. Distance only counts while the box is growing (finite TTC),
    so vehicles sitting close in a jam don't keep the light on.
    """
    closing = np.isfinite(ttc_s)                                  # growing box = approaching
    red = (ttc_s < RED_TTC) | (closing & (dist_m < RED_DIST))
    amber = (ttc_s < AMBER_TTC) | (closing & (dist_m < AMBER_DIST))
    levels = np.select([red, amber], [2, 1], default=0)
    return pd.Series(levels, index=ttc_s.index)


def frame_levels(df):
    """One row per frame: the highest level on the rider's left and right.

    CENTRE boxes count for BOTH sides: a vehicle right behind can pass on either side.
    """
    df = df.copy()
    df["level"] = box_level(df["ttc_s"], df["dist_m"])
    df["left"] = np.where(df["zone"].isin(["LEFT", "CENTRE"]), df["level"], 0)
    df["right"] = np.where(df["zone"].isin(["RIGHT", "CENTRE"]), df["level"], 0)
    levels = df.groupby(["frame", "t"], as_index=False)[["left", "right"]].max()
    return levels.rename(columns={"left": "left_level", "right": "right_level"})

def fill_missing_frames(levels):
    """Add a row (both lights 0) for every analysed frame that had no boxes, so 1 row = 1 step in time."""
    step = int(levels["frame"].diff().min())          # smallest gap between rows = the analysis step
    all_frames = range(levels["frame"].min(), levels["frame"].max() + 1, step)
    full = levels.set_index("frame").reindex(all_frames).rename_axis("frame")
    cols = ["left_level", "right_level"]
    full[cols] = full[cols].fillna(0).astype(int)     # new rows: no boxes, so lights off
    full["t"] = full["t"].interpolate()               # fill t in the new rows from their neighbours
    return full.reset_index()


def apply_hold(levels, fps=10.0, hold_s=HOLD_S):
    """Keep each light on for at least hold_s after it was last triggered (removes flicker).

    Expects one row per analysed frame (run fill_missing_frames first).
    """
    hold = max(1, round(hold_s * fps))                 # 0.5 s * 10 fps = 5 rows
    out = levels.copy()
    for col in ["left_level", "right_level"]:
        out[col] = out[col].rolling(hold, min_periods=1).max().astype(int)
    return out


def draw_lights(frame, left, right):
    """Draw the rider's two lights at the top of the frame: 0 grey, 1 amber, 2 red.

    Mirror: each light sits above the image part it watches. Rider's RIGHT light (R) at the
    image's top-left, rider's LEFT light (L) at the image's top-right.
    """
    colours = {0: GREY, 1: AMBER, 2: RED}
    h, w = frame.shape[:2]
    r = max(18, w // 40)                                   # radius scales with the frame width
    for level, x, label in [(right, 2 * r, "R"), (left, w - 2 * r, "L")]:
        cv2.circle(frame, (x, 2 * r), r, colours[level], thickness=-1)     # -1 = filled
        cv2.putText(frame, label, (x - r // 3, 2 * r + r // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 30, (255, 255, 255), 2)
    return frame


def render(ride_dir, levels, geo, width=1280):
    """Write video_rules.mp4: every analysed frame with zone lines, boxes + TTC, and the two lights."""
    ride_dir = Path(ride_dir)
    lights = {f: (l, r) for f, l, r in zip(levels["frame"], levels["left_level"], levels["right_level"])}
    boxes = {f: rows for f, rows in geo.rename(columns={"class": "cls"}).groupby("frame")}

    cap = cv2.VideoCapture(str(ride_dir / "video.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / 10))                     # same step as track.py
    out = ride_dir / "video_rules.mp4"
    writer, frame_idx = None, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            if frame.shape[1] > width:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

            for x in (round(w * SIDE_FRACTION), round(w * (1 - SIDE_FRACTION))):    # zone borders
                cv2.line(frame, (x, 0), (x, h - 1), (200, 200, 200), 1)

            rows = boxes.get(frame_idx)
            if rows is not None:
                for r in rows.itertuples():
                    ttc = "" if np.isinf(r.ttc_s) else f" {r.ttc_s:.1f}s"
                    draw(frame, r.x1, r.y1, r.x2, r.y2, f"#{r.track_id}{ttc}",
                            COLOURS.get(r.cls, (255, 255, 255)))

            left, right = lights.get(frame_idx, (0, 0))
            draw_lights(frame, left, right)
            cv2.putText(frame, f"t={frame_idx / src_fps:5.1f}s", (w // 2 - 70, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / step, (w, h))
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    ap.add_argument("--video", action="store_true", help="also write video_rules.mp4")
    args = ap.parse_args()

    ride = Path(args.ride_dir)
    geo = pd.read_parquet(ride / "video_geometry.parquet")
    levels = apply_hold(fill_missing_frames(frame_levels(geo)))
    levels.to_parquet(ride / "video_rules.parquet", index=False)
    on = (levels[["left_level", "right_level"]] > 0).mean()
    print(f"Light on: left {on['left_level']:.0%} of frames, right {on['right_level']:.0%} of frames")
    minutes = (levels["t"].max() - levels["t"].min()) / 60
    for col in ["left_level", "right_level"]:
        red = levels[col] == 2
        starts = (red & ~red.shift(fill_value=False)).sum()           # red now, not red one row before
        print(f"{col}: {starts} red alarms, {starts / minutes:.1f} per minute")
    if args.video:
            render(ride, levels, geo)
