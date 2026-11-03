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


def processed_size(meta, width=WIDTH):
    """Frame size the boxes were measured in (after track.py's resize)."""
    w, h = meta["width"], meta["height"]
    if w > width:
        return width, round(h * width / w)
    return w, h


def zone_of(cx, frame_w):
    """Rider's side for a box centre x: "RIGHT", "CENTRE" or "LEFT" (a pandas Series in, Series out).

    The camera faces BACKWARDS, so the image is mirrored compared with the rider's view:
    the LEFT third of the image is the rider's RIGHT side.

    TODO:
      1. Split the frame width into thirds (start with thirds, tune the borders later by eye).
      2. Use np.select([cx < frame_w / 3, cx > 2 * frame_w / 3], ["RIGHT", "LEFT"], default="CENTRE")
      3. Return it as a pd.Series with the same index as cx.

    Check yourself: India drives on the left. On which side of the IMAGE should most overtakes appear?
    """
    raise NotImplementedError("TODO: zone_of")


def add_ttc(df, fps=FPS, window_s=TTC_WINDOW_S):
    """Add `ttc_s`: seconds until the object reaches you, from how fast its box grows.

    Idea: if the box height h grows, the object is getting closer. For a steady approach,
    log(h) rises in a straight line, and TTC = 1 / slope_of_log_h_per_second.
    Shrinking or steady boxes (slope <= 0) never reach you: TTC = inf.

    TODO:
      1. df = df.sort_values(["track_id", "frame"]); df["log_h"] = np.log(df["h"])
      2. n = max(3, round(window_s * fps)) frames per window.
      3. Write  def slope(values):  that fits a straight line to `values` against time
         (np.arange(len(values)) / fps) with np.polyfit(..., 1) and returns the slope.
      4. Per track, rolling over the last n rows, apply `slope`:
            df.groupby("track_id")["log_h"].rolling(n, min_periods=3).apply(slope, raw=True)
         The result has an extra index level: .reset_index(level=0, drop=True) lines it up with df.
      5. ttc = 1 / slope where slope > 0, else np.inf. Clip very large values to np.inf (e.g. > 20 s).
      6. Drop the helper column, return df.

    Check yourself: rolling windows look BACKWARDS by default. Why does that matter later,
    when this TTC becomes a feature for the model in features.py?
    """
    raise NotImplementedError("TODO: add_ttc")


def add_distance(df, calib):
    """Add `dist_m` from where the box touches the road: Z = k / (y2 - y_horizon).

    TODO:
      1. If calib is None: df["dist_m"] = np.nan and return (calibration comes later).
      2. Otherwise compute k / (df["y2"] - calib["y_horizon"]).
      3. Boxes whose bottom is at or above the horizon (y2 <= y_horizon) have no valid distance: NaN.
      4. Very large values (> 50 m) aren't trustworthy either: NaN.

    Check yourself: why the BOTTOM edge (y2) and not the box centre?
    """
    raise NotImplementedError("TODO: add_distance")


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
    print(df[["zone", "ttc_s", "dist_m"]].describe(include="all").to_string())
    print(f"Saved {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    geometry(ap.parse_args().ride_dir)
