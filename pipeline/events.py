"""Automatic labels for the alert model: did this vehicle come dangerously close soon after?

Supervised learning needs "correct answers". At time t nobody knows what a vehicle will do next,
but the RECORDING does: look at the next HORIZON_S seconds of the same track and decide.
These are near-miss labels (surrogate safety measures): crashes can't be recorded on purpose.

    label = 1  if, within the next HORIZON_S seconds, the track has a DANGER EVENT:
                 A. closer than its class's danger distance AND inside the corridor  (needs calibration)
                 B. passed alongside: its last row is big and touches the left or right edge
                 C. too close to measure: big and touching the bottom edge
    label = 0  otherwise

B and C need no calibration. A needs data/calibration.json, so that geometry.py has filled
dist_m and in_corridor (re-run geometry.py after calibrating). Until then A finds nothing.

Pipeline position:  geometry.py -> events.py -> features.py -> risk.py

Input : data/rides/<ride>/video_geometry.parquet, meta.json
Output: data/rides/<ride>/video_events.parquet   geometry rows + label

Usage:
    uv run python pipeline/events.py data/rides/2026-11-20_1

Done when: you've watched the positive moments and written down how many were real close
passes (e.g. "44/50 correct"). That number goes in the README.

ML ideas you'll practise: label design, the label looks into the FUTURE but features must not
(leakage), class imbalance (positives will be rare).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geometry 
from geometry import processed_size
HORIZON_S = 2.0
WIDTH = 1280
EDGE_MARGIN = 0.03   # within 3% of the left/right/bottom image edge counts as "at the edge"
BIG_H = 0.25         # box taller than 25% of the frame height counts as "big" (close)

# Starting values from riding experience (NOTES section 11): two-wheelers are agile and come
# close routinely; heavier vehicles have worse driver visibility and hit harder.
DANGER_DIST_M = {"two_wheeler": 2.0, "person": 2.0, "animal": 2.0,
                 "car": 4.0, "auto_rickshaw": 4.0, "other": 4.0, "bus_truck": 6.0}


def close_events(track):
    """Event A: times when the track was within its class's danger distance AND in the corridor.

    Returns an empty array until calibrated (dist_m is NaN, and NaN < x is False).

    TODO:
      1. limit = DANGER_DIST_M.get(track["class"].iloc[0], 4.0)     # unknown class -> 4 m
      2. hit = (track["dist_m"] < limit) & track["in_corridor"]
      3. return track.loc[hit, "t"].to_numpy()
    """
    raise NotImplementedError("TODO: close_events")


def edge_events(track, frame_w, frame_h):
    """Events B and C, which need no calibration.

    B. passed alongside: the track's LAST row is big and touches the left or right edge.
    C. too close to measure: any row that is big and touches the bottom edge.
    Returns a numpy array of event times (seconds).
    """
    h = track["y2"] - track["y1"]                                   # box height of every row
    big = h > BIG_H * frame_h                                        # taller than 25% of the frame

    left = track["x1"] < EDGE_MARGIN * frame_w                              # touching the left edge
    right = track["x2"] > (1 - EDGE_MARGIN) * frame_w                       # touching the right edge
    side = left | right                                           # either edge
    bottom = track["y2"] > (1 - EDGE_MARGIN) * frame_h             # touching the bottom edge

    passed = (big & side).iloc[-1]                                 # only the LAST row counts for B
    times_b = track["t"].to_numpy()[-1:] if passed else np.array([])
    times_c = track.loc[big & bottom, "t"].to_numpy()
    return np.concatenate([times_b, times_c])


def label_track(track, frame_w, frame_h, horizon=HORIZON_S):
    """0/1 Series (same index as `track`): is there a danger event between t and t + horizon?

    TODO:
      1. events = np.concatenate([close_events(track), edge_events(track, frame_w, frame_h)])
      2. If there are no events: return pd.Series(0, index=track.index)
      3. times = track["t"].to_numpy()
         ahead = (events[None, :] >= times[:, None]) & (events[None, :] <= times[:, None] + horizon)
         This is a (rows x events) True/False table, all pairs at once (broadcasting).
         A row is positive if ANY event lies ahead of it:  label = ahead.any(axis=1)
      4. return pd.Series(label.astype(int), index=track.index)

    Check yourself: the label at time t uses data from AFTER t. Why is that fine for a LABEL,
    but would be cheating if the same future data were used as a FEATURE?
    """
    raise NotImplementedError("TODO: label_track")


def events(ride_dir):
    ride_dir = Path(ride_dir)
    meta = json.loads((ride_dir / "meta.json").read_text())
    w, h = processed_size(meta)
    frame_w, frame_h = (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)

    df = pd.read_parquet(ride_dir / "video_geometry.parquet").sort_values(["track_id", "frame"])
    calibrated = df["dist_m"].notna().any()
    df["label"] = (df.groupby("track_id", group_keys=False)
                     .apply(lambda tr: label_track(tr, frame_w, frame_h)))
    df.to_parquet(ride_dir / "video_events.parquet", index=False)

    print(f"{len(df)} rows, {df['label'].sum()} positive ({df['label'].mean():.1%}).")
    print("Tracks with at least one positive:", df.loc[df["label"] == 1, "track_id"].nunique())
    print(df.groupby("class")["label"].mean().round(3).to_string())
    if not calibrated:
        print("NOTE: no calibration yet -> only events B (alongside) and C (bottom edge) were used.")
    print("Next: watch the positive moments and count how many are real close passes.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    events(ap.parse_args().ride_dir)
