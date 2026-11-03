"""Fit the "pixel row -> metres behind the scooter" formula from tape marks on the road.

How to take the calibration frame (once, mount in its usual taped position):
    1. Park on flat ground. Put tape lines across the road 2, 4, 6 and 10 m behind the camera.
    2. Record a 5-second video, ingest it like a ride.
    3. Run this script on it, click each tape line, type its distance.

Why it works: on flat ground, something touching the road further away appears higher in the
image, closer to the horizon. The relationship is  Z = k / (y - y_horizon)  where y is the pixel
row where the object touches the road. Two unknowns (k, y_horizon), so 2 marks is the minimum;
4 marks let you check the fit.

Input : data/rides/<calibration ride>/video.mp4
Output: data/calibration.json  {"k": ..., "y_horizon": ..., "frame_w": ..., "frame_h": ..., "rms_error_m": ...}

Usage:
    uv run python pipeline/calibrate.py data/rides/2026-11-07_1/video.mp4

Done when: calibration.json exists and rms_error_m is under ~0.5 m.

Python you'll practise: OpenCV windows and mouse callbacks, numpy polyfit, JSON.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
WIDTH = 1280  # must match track.py, so pixel rows mean the same thing


def first_frame(video, width=WIDTH):
    """Return the first frame, resized exactly like track.py does."""
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Can't read {video}")
    if frame.shape[1] > width:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    return frame


def click_rows(frame):
    """Show the frame; return the pixel rows (y values) the user clicks, in click order.

    TODO:
      1. Make an empty list `rows`.
      2. Write a callback  def on_mouse(event, x, y, flags, param):  that appends y to `rows`
         when event == cv2.EVENT_LBUTTONDOWN, and draws a horizontal line at y on a copy of the frame.
      3. cv2.namedWindow("calibrate"); cv2.setMouseCallback("calibrate", on_mouse)
      4. Loop: cv2.imshow(...); key = cv2.waitKey(20); stop when key is Enter (13) or Esc (27).
      5. cv2.destroyAllWindows(); return rows

    Check yourself: why does the callback need to change a list defined outside it,
    and why can it append to `rows` without the `nonlocal` keyword?
    """
    raise NotImplementedError("TODO: click_rows")


def fit(rows, distances):
    """Fit Z = k / (y - y_horizon). Return (k, y_horizon).

    The trick (linearising): flip both sides.  1/Z = (y - y_horizon) / k = (1/k) * y - y_horizon/k
    That is a straight line in y:  1/Z = a*y + b,  with a = 1/k and b = -y_horizon/k.

    TODO:
      1. y = np.array(rows, dtype=float); inv_z = 1 / np.array(distances, dtype=float)
      2. a, b = np.polyfit(y, inv_z, 1)          # straight-line fit, returns [slope, intercept]
      3. k = 1 / a;  y_horizon = -b / a
      4. return k, y_horizon

    Check yourself: why fit 1/Z instead of Z directly? (Hint: polyfit only fits straight lines.)
    """
    raise NotImplementedError("TODO: fit")


def rms_error(rows, distances, k, y_horizon):
    """Root-mean-square error in metres between the formula and the tape measurements.

    TODO: predicted = k / (np.array(rows) - y_horizon); return sqrt(mean((predicted - distances)**2))
    """
    raise NotImplementedError("TODO: rms_error")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    args = ap.parse_args()

    frame = first_frame(args.video)
    print("Click each tape line where it touches the road, then press Enter.")
    rows = click_rows(frame)
    distances = [float(input(f"Distance in metres for the line at row {y}: ")) for y in rows]

    k, y_horizon = fit(rows, distances)
    err = rms_error(rows, distances, k, y_horizon)
    result = {"k": k, "y_horizon": y_horizon, "frame_w": frame.shape[1],
              "frame_h": frame.shape[0], "rms_error_m": err}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved {OUT}")
