"""Turn tracked boxes into what the alert needs: which side, how far, and how soon.

Pipeline position:  track.py -> clean_tracks.py -> geometry.py -> rules.py / events.py

Input : data/rides/<ride>/video_tracks_clean.parquet  (frame, t, track_id, class, conf, x1, y1, x2, y2)
        data/rides/<ride>/meta.json                   (video width/height)
        data/calibration.json                         (optional until you've run calibrate.py)
Output: data/rides/<ride>/video_geometry.parquet      same rows + cx, h, zone, dist_m, ttc_s

Usage:
    uv run python pipeline/geometry.py data/rides/2026-11-03_2

Done when: on the overlay of a ride, a vehicle catching up shows a TTC that falls towards 0,
and a vehicle falling behind shows TTC = inf.

Python you'll practise: small pure functions, numpy on whole columns at once, pandas groupby + rolling.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATION = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
WIDTH = 1280        # track.py resizes to at most this width; boxes are in that size
FPS = 10.0          # track.py analyses 10 frames per second
TTC_WINDOW_S = 0.5  # fit the growth over the last 0.5 s (5 frames)
SIDE_FRACTION = 1 / 3   # each side zone covers this share of the frame width
MAX_TTC_S = 20.0  # TTC above this is "no threat": stored as inf
MAX_DIST_M = 50.0  # beyond this, the ground-contact distance is too uncertain: NaN
HALF_WIDTH_M = 0.45   # Activa mirror-tip half-width; confirm with a tape
CORRIDOR_M   = 1.0     # safety buffer each side; tune on the judged alarms once calibrated

def processed_size(meta, width=WIDTH):
    """Frame size the boxes were measured in (after track.py's resize)."""
    w, h = meta["width"], meta["height"]
    if w > width:
        return width, round(h * width / w)
    return w, h


def zone_of(cx, frame_w, side=SIDE_FRACTION):
    """Rider's side for each box centre x: "RIGHT", "CENTRE" or "LEFT".

    The camera faces backwards, so the image is mirrored:
    image-left = rider's RIGHT, image-right = rider's LEFT.
    """
    left_edge = frame_w * side            # pixels: end of the image-left zone
    right_edge = frame_w * (1 - side)     # pixels: start of the image-right zone

    in_image_left = cx < left_edge             # True/False for every row
    in_image_right = cx > right_edge

    zones = np.select(
        [in_image_left, in_image_right],     # conditions, checked in order
        ["RIGHT", "LEFT"], # what each one means for the RIDER (mirror!)
        default="CENTRE",                    # neither condition is true
    )
    return pd.Series(zones, index=cx.index)


def add_ttc(df, fps=FPS, window_s=TTC_WINDOW_S, max_ttc=MAX_TTC_S):
    """Add `ttc_s`: seconds until the object reaches you, from how fast its box grows.

    For a steady approach, ln(h) grows at rate 1/TTC, so TTC = 1 / slope of ln(h).
    Compares each box with the SAME track's box `window_s` earlier (past only).
    Shrinking/steady boxes, or tracks without enough history yet: TTC = inf.
    """
    df = df.sort_values(["track_id", "frame"]).copy()
    n = max(1, round(window_s * fps))                  # rows to look back (0.5 s * 10 fps = 5)

    log_h = np.log(df["h"])
    by_track = df["track_id"]

    d_log_h = log_h - log_h.groupby(by_track).shift(n)      # growth over the last n rows
    d_t = df["t"] - df["t"].groupby(by_track).shift(n)      # real seconds between them

    slope = d_log_h / d_t                              # growth rate per second (1/s)
    df["ttc_s"] = np.where(slope > 0, 1 / slope, np.inf)
    df.loc[df["ttc_s"] > max_ttc, "ttc_s"] = np.inf
    return df


def add_distance(df, calib, max_dist=MAX_DIST_M):
    """Add `dist_m` from where the box touches the road: Z = k / (y2 - y_horizon).

    calib is None until calibrate.py has written data/calibration.json: dist_m stays NaN.
    """
    df = df.copy()
    df["dist_m"] = np.nan
    if calib is None:
        return df

    below = df["y2"] - calib["y_horizon"]          # pixels between the box bottom and the horizon
    valid = below > 0                      # at or above the horizon: no ground contact, no distance
    df.loc[valid, "dist_m"] = calib["k"] / below[valid]
    df.loc[df["dist_m"] > max_dist, "dist_m"] = np.nan
    return df


def geometry(ride_dir):
    ride_dir = Path(ride_dir)
    meta = json.loads((ride_dir / "meta.json").read_text())
    src = ride_dir / "video_tracks_clean.parquet"
    if not src.exists():  # fall back to raw tracks until clean_tracks.py is done
        src = ride_dir / "video_tracks.parquet"
    df = pd.read_parquet(src)
    calib = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else None

    frame_w, _ = processed_size(meta)
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["h"] = df["y2"] - df["y1"]
    df["zone"] = zone_of(df["cx"], frame_w)
    df = add_ttc(df)
    df = add_distance(df, calib)

    out = ride_dir / "video_geometry.parquet"
    df.to_parquet(out, index=False)
    finite = df["ttc_s"].replace(np.inf, np.nan)
    print(df["zone"].value_counts().to_string())
    print(f"rows with a finite TTC: {finite.notna().mean():.0%}")
    print(finite.describe().round(2).to_string())
    print(f"rows with a distance: {df['dist_m'].notna().mean():.0%}")
    return df

def draw_reference(frame, calib, frame_w):
    """Overlay the horizon, the collision corridor, and (if k is known) distance rings."""
    import cv2
    h, w = frame.shape[:2]
    yh, H = int(calib["y_horizon"]), calib["camera_height"]
    c0 = calib.get("c0", frame_w / 2)
    grey, amber = (200, 200, 200), (0, 191, 255)
    cv2.line(frame, (0, yh), (w, yh), grey, 1)                       # horizon

    half = HALF_WIDTH_M + CORRIDOR_M                                 # corridor half-width in metres
    for sign in (-1, 1):                                             # left and right edge of the corridor
        pts = [(int(c0 + sign * half * (y - yh) / H), y) for y in range(yh + 1, h, 8)]
        cv2.polylines(frame, [np.array(pts)], False, amber, 2)

    if "k" in calib:                                                 # exact rings only once calibrated
        for Z in (3, 5, 10, 20):
            y = yh + round(calib["k"] / Z)
            if yh < y < h:
                cv2.line(frame, (0, y), (w, y), grey, 1)
                cv2.putText(frame, f"{Z} m", (10, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grey, 1)
    if calib.get("approx"):
        cv2.putText(frame, "CORRIDOR APPROX (uncalibrated)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, amber, 2)
    return frame

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    geometry(ap.parse_args().ride_dir)
