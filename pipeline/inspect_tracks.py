"""Summarise every track in a ride and flag the suspicious ones (likely false detections).

Usage:
    uv run python pipeline/inspect_tracks.py data/rides/2026-11-03_2
Writes data/rides/<ride>/track_summary.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

WIDTH = 1280      # track.py's resize limit, so we know the frame size the boxes are in
WIDE = 0.80       # wider than 80% of the frame: suspicious
EDGE = 0.02       # within 2% of the left/right edge counts as "touching"
SHORT_S = 0.5     # tracks shorter than this are flickers
LOW_CONF = 0.5


def frame_size(ride):
    meta = json.loads((ride / "meta.json").read_text())
    w, h = meta["width"], meta["height"]
    return (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)


def summarise(df, frame_w, frame_h):
    """One row per track: class, length, confidence, size and edge contact."""
    df = df.assign(
        w=(df.x2 - df.x1) / frame_w,                    # box width as a fraction of the frame
        h=(df.y2 - df.y1) / frame_h,
        both_edges=(df.x1 < EDGE * frame_w) & (df.x2 > (1 - EDGE) * frame_w),
    )
    s = df.groupby("track_id").agg(
        cls=("class", lambda c: c.mode()[0]),           # most frequent class
        n_classes=("class", "nunique"),                 # >1 means it switched class
        frames=("frame", "count"),
        start=("t", "min"),
        end=("t", "max"),
        conf=("conf", "mean"),
        w=("w", "mean"),
        h=("h", "mean"),
        both_edges=("both_edges", "mean"),              # share of frames touching both edges
    )
    s["seconds"] = s["end"] - s["start"]
    return s


def flag(s):
    """A text column listing every reason a track looks suspicious."""
    reasons = pd.Series("", index=s.index)
    checks = {
        "too_wide": s["w"] > WIDE,
        "both_edges": s["both_edges"] > 0.5,
        "flicker": s["seconds"] < SHORT_S,
        "low_conf": s["conf"] < LOW_CONF,
        "class_switch": s["n_classes"] > 1,
    }
    for name, mask in checks.items():
        reasons[mask] = reasons[mask] + name + " "
    return reasons.str.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ride_dir")
    ride = Path(ap.parse_args().ride_dir)

    tracks = pd.read_parquet(ride / "video_tracks.parquet")
    summary = summarise(tracks, *frame_size(ride))
    summary["flags"] = flag(summary)
    summary.round(2).to_csv(ride / "track_summary.csv")

    suspicious = summary[summary["flags"] != ""].sort_values("frames", ascending=False)
    print(f"{len(summary)} tracks, {len(suspicious)} suspicious\n")
    print(suspicious.round(2).to_string())