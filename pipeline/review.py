"""Step through a ride with its tracked boxes, tune the filters live, and mark mistakes.

Usage:
    uv run python pipeline/review.py data/rides/2026-11-03_2

Keys (click the video window first):
    space   play / pause
    d / a   next / previous frame
    m       mark this frame as wrong; type a note in the terminal, then press Enter
    q, Esc  quit
Sliders:
    min conf %    hide boxes below this confidence
    min height %  hide boxes shorter than this % of the frame height (far away)

Marks are appended to data/rides/<ride>/review_marks.csv (frame, t, note).
"""
import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

from detect import COLOURS, draw

WIN = "review"
WIDTH = 1280   # same resize as track.py, so the boxes line up
FPS = 10.0     # track.py analyses 10 frames per second


def read_frame(cap, n, width=WIDTH):
    """Frame number n of the video, resized exactly like track.py does."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[1] > width:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return frame


def save_mark(path, n, t, note):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["frame", "t", "note"])
        writer.writerow([n, round(t, 2), note])


def review(ride):
    ride = Path(ride)
    tracks = pd.read_parquet(ride / "video_tracks.parquet").rename(columns={"class": "cls"})
    by_frame = {n: rows for n, rows in tracks.groupby("frame")}
    marks = ride / "review_marks.csv"

    cap = cv2.VideoCapture(str(ride / "video.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / FPS))
    frame_ids = list(range(0, total, step))  # the frames track.py analysed

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # resizable, handy for tall portrait videos
    cv2.createTrackbar("min conf %", WIN, 40, 100, lambda v: None)
    cv2.createTrackbar("min height %", WIN, 6, 30, lambda v: None)

    i, playing, cached_n, cached = 0, False, None, None
    while 0 <= i < len(frame_ids):
        n = frame_ids[i]
        if n != cached_n:  # only decode when the frame changes (sliders redraw from the cache)
            cached, cached_n = read_frame(cap, n), n
            if cached is None:
                break
        frame = cached.copy()

        conf_min = cv2.getTrackbarPos("min conf %", WIN) / 100
        h_min = cv2.getTrackbarPos("min height %", WIN) / 100 * frame.shape[0]
        shown = 0
        rows = by_frame.get(n)
        if rows is not None:
            for r in rows.itertuples():
                if r.conf < conf_min or (r.y2 - r.y1) < h_min:
                    continue
                draw(frame, r.x1, r.y1, r.x2, r.y2, f"#{r.track_id} {r.cls} {r.conf:.2f}",
                     COLOURS.get(r.cls, (255, 255, 255)))
                shown += 1

        status = f"frame {n}  t={n / src_fps:5.1f}s  boxes {shown}  {'PLAY' if playing else 'PAUSED'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            i, playing = i + 1, False
        elif key == ord("a"):
            i, playing = max(0, i - 1), False
        elif key == ord("m"):
            playing = False
            note = input(f"Note for frame {n} (t={n / src_fps:.1f}s): ")
            save_mark(marks, n, n / src_fps, note)
            print(f"Saved to {marks}")
        elif playing:
            i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    review(ap.parse_args().ride_dir)