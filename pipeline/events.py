"""Automatic labels for the alert model: did this object pass dangerously close soon after?

This is where supervised learning gets its "correct answers" without anyone labelling by hand.
At time t you can't know what an object will do. But the RECORDING knows: look at the next
2 seconds of the same track and decide.

    label = 1  if, within the next HORIZON_S seconds, the track
                  - comes closer than CLOSE_M metres, or
                  - leaves the frame at the left/right edge while its box is big (it passed alongside you)
    label = 0  otherwise

Pipeline position:  geometry.py -> events.py -> features.py -> risk.py

Input : data/rides/<ride>/video_geometry.parquet, meta.json
Output: data/rides/<ride>/video_events.parquet   geometry rows + label

Usage:
    uv run python pipeline/events.py data/rides/2026-11-03_2

Done when: you've watched ~50 positive moments in the tracked video and written down how many
were real close passes (e.g. "44/50 correct"). That number goes in the README.

ML ideas you'll practise: label design, the label "looks into the future" but the features
must NOT (leakage), class imbalance (positives will be rare).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_S = 2.0
CLOSE_M = 2.0
EDGE_MARGIN = 0.03   # box within 3% of the left/right image edge counts as "at the edge"
BIG_H = 0.25         # box taller than 25% of the frame counts as "big" (close)
WIDTH = 1280


def passed_alongside(track, frame_w, frame_h):
    """True for the LAST row of a track that ended big and touching a side edge.

    TODO:
      1. last = track.iloc[-1]
      2. at_edge = last.x1 < EDGE_MARGIN * frame_w  or  last.x2 > (1 - EDGE_MARGIN) * frame_w
      3. big = (last.y2 - last.y1) > BIG_H * frame_h
      4. return the time last.t if at_edge and big, else None
    """
    raise NotImplementedError("TODO: passed_alongside")


def label_track(track, frame_w, frame_h):
    """Return a 0/1 Series (same index as `track`): is there a close event within HORIZON_S ahead?

    TODO:
      1. Event times for this track:
           - every t where dist_m < CLOSE_M
           - the time from passed_alongside(...) if not None
      2. For each row time t: label 1 if any event time e has t <= e <= t + HORIZON_S.
         (numpy: compare the column of row times against the array of event times; broadcasting
          with times[:, None] and events[None, :] does all pairs at once.)
      3. Return a pd.Series of ints with track.index.

    Check yourself: the label at time t uses data from AFTER t. Why is that fine for a label,
    but would be cheating if the same future data were used as a feature?
    """
    raise NotImplementedError("TODO: label_track")


def events(ride_dir):
    ride_dir = Path(ride_dir)
    meta = json.loads((ride_dir / "meta.json").read_text())
    w, h = meta["width"], meta["height"]
    frame_w, frame_h = (WIDTH, round(h * WIDTH / w)) if w > WIDTH else (w, h)

    df = pd.read_parquet(ride_dir / "video_geometry.parquet").sort_values(["track_id", "frame"])
    df["label"] = (df.groupby("track_id", group_keys=False)
                     .apply(lambda tr: label_track(tr, frame_w, frame_h)))
    df.to_parquet(ride_dir / "video_events.parquet", index=False)

    pos = df["label"].mean()
    print(f"{len(df)} rows, {df['label'].sum()} positive ({pos:.1%}).")
    print("Tracks with at least one positive:", df.loc[df["label"] == 1, "track_id"].nunique())
    print("Next: watch those moments in video_tracked.mp4 and count how many are real close passes.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    events(ap.parse_args().ride_dir)
