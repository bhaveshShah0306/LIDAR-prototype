"""Clean raw tracks: drop flickers, one class per track, smooth boxes.

Usage:
    uv run python pipeline/clean_tracks.py data/rides/2026-11-12_1
Reads video_tracks.parquet, writes video_tracks_clean.parquet next to it.
"""
import argparse
from pathlib import Path

import pandas as pd

MIN_SECONDS = 0.5   # shorter tracks are flicker, not real objects
SMOOTH_FRAMES = 5   # rolling-median window for box edges


def drop_short(df):
    """Keep only tracks that last at least MIN_SECONDS."""
    start = df.groupby("track_id")["t"].transform("min")      # first time of each row's track
    end = df.groupby("track_id")["t"].transform("max")        # last time of each row's track
    duration = end - start                                  # one duration per row
    return df[duration >= MIN_SECONDS].copy()                              # keep rows of long-enough tracks


def majority_class(df):
    """Give every row of a track that track's most frequent class.
    Hint: groupby("track_id")["class"].transform(lambda s: s.mode()[0])
    """
    df = df.copy()
    df["class_raw"]= df["class"]
    df["class"] = df.groupby("track_id")["class"].transform(lambda s: s.mode()[0])
    return df

def smooth_boxes(df):
    """Rolling median of x1, y1, x2, y2 within each track (sort by track_id, frame first!).
    Hint: groupby("track_id")[cols].transform(lambda s: s.rolling(SMOOTH_FRAMES, min_periods=1, center=True).median())
    Check yourself: center=True uses future frames. Fine for cleaning recorded rides,
    but what would it mean for the live demo?
    """
    df = df.sort_values(["track_id", "frame"]).copy()          # time order inside each track
    cols = ["x1", "y1", "x2", "y2"]
    df[cols] = df.groupby("track_id")[cols].transform(
        lambda s: s.rolling(SMOOTH_FRAMES, min_periods=1, center=True).median()
    )
    return df


def summary(df, title):
    """Print boxes and tracks per class."""
    s = df.groupby("class").agg(boxes=("frame", "size"), tracks=("track_id", "nunique"))
    print(f"\n{title}\n{s.to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ride_dir")
    ride = Path(ap.parse_args().ride_dir)
    raw = pd.read_parquet(ride / "video_tracks.parquet")
    summary(raw, "raw")
    clean = smooth_boxes(majority_class(drop_short(raw)))
    summary(clean, "clean")
    clean.to_parquet(ride / "video_tracks_clean.parquet", index=False)
    print(f"Saved {ride / 'video_tracks_clean.parquet'}")