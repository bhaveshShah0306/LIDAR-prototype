"""Find UNIDENTIFIED objects: anything growing in the image that YOLO didn't box.

The camera faces backwards while you ride forwards, so the world you're leaving behind
(road, parked cars, poles) SHRINKS towards the vanishing point. Anything catching up with you
GROWS. Optical flow measures how every pixel moves between two frames; where the flow spreads
outwards (positive divergence), something is getting closer, whatever it is.

Pipeline position:  track.py (YOLO boxes) + video -> looming.py -> geometry/features as class "unknown"

Input : data/rides/<ride>/video.mp4, data/rides/<ride>/video_tracks.parquet (to skip YOLO's boxes)
Output: data/rides/<ride>/video_unknown.parquet   frame, t, x1, y1, x2, y2, ttc_s
        data/rides/<ride>/video_looming.mp4       (with --video) divergence heat map + unknown boxes

Usage:
    uv run python pipeline/looming.py data/rides/2026-11-03_2 --video

Done when: on video_looming.mp4 the background stays dark, and a vehicle catching up lights up.

Python you'll practise: numpy arrays as images, OpenCV optical flow, gradients, connected components.
Work at HALF resolution (640 wide): flow is expensive, and you only need coarse blobs.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORK_WIDTH = 640
FPS = 10.0
MIN_DIVERGENCE = 0.04   # per frame; tune by looking at the heat map
MIN_AREA = 400          # pixels at 640 wide; smaller blobs are noise


def stabilise(prev_gray, gray):
    """Cancel phone shake: warp prev_gray so its background lines up with gray.

    TODO:
      1. p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8)
      2. p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
      3. Keep pairs where status == 1. If fewer than ~20 remain, return prev_gray unchanged.
      4. H, _ = cv2.findHomography(p0_good, p1_good, cv2.RANSAC, 3.0)
         RANSAC ignores points on moving vehicles and fits the dominant (background) motion.
      5. return cv2.warpPerspective(prev_gray, H, (width, height))

    Check yourself: what would happen to the divergence map if you skipped this step on a bumpy road?
    """
    raise NotImplementedError("TODO: stabilise")


def flow_divergence(prev_gray, gray, dis):
    """Dense flow between two frames, then its divergence (how much it spreads out at each pixel).

    TODO:
      1. flow = dis.calc(prev_gray, gray, None)          # H x W x 2: (dx, dy) per pixel
      2. fx, fy = flow[..., 0], flow[..., 1]
      3. div = np.gradient(fx, axis=1) + np.gradient(fy, axis=0)
      4. Smooth it: cv2.GaussianBlur(div, (0, 0), 3)    # removes speckle
      5. return div
    """
    raise NotImplementedError("TODO: flow_divergence")


def growing_blobs(div, yolo_boxes, scale):
    """Boxes around strongly growing regions that are NOT already inside a YOLO box.

    `yolo_boxes` are in full processed coordinates; multiply by `scale` to get work coordinates.

    TODO:
      1. mask = (div > MIN_DIVERGENCE).astype(np.uint8)
      2. For each YOLO box, set that rectangle of the mask to 0 (YOLO already knows about it).
      3. n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
      4. For each component except 0 (the background) with stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
         x, y, w, h from stats -> box; mean divergence inside -> ttc_s = 2 / mean_div / FPS
         (pure expansion: divergence per frame = 2 / frames-to-contact)
      5. Return a list of [x1, y1, x2, y2, ttc_s] scaled back to full coordinates (divide by scale).
    """
    raise NotImplementedError("TODO: growing_blobs")


def looming(ride_dir, save_video=False):
    ride_dir = Path(ride_dir)
    tracks = pd.read_parquet(ride_dir / "video_tracks.parquet")
    boxes_by_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in tracks.groupby("frame")}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    cap = cv2.VideoCapture(str(ride_dir / "video.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / FPS))
    rows, prev, frame_idx = [], None, 0
    # TODO: the same reading loop as track.py, for every `step`-th frame:
    #   1. resize to the processed width (1280) like track.py, then again to WORK_WIDTH;
    #      scale = WORK_WIDTH / processed_width
    #   2. gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    #   3. if prev is not None: div = flow_divergence(stabilise(prev, gray), gray, dis)
    #                           for each blob in growing_blobs(div, boxes_by_frame.get(frame_idx, []), scale):
    #                               rows.append([frame_idx, frame_idx / src_fps, *blob])
    #   4. prev = gray
    #   5. if save_video: draw div as a heat map (cv2.applyColorMap on a 0..255 version) + blob boxes
    raise NotImplementedError("TODO: reading loop")

    df = pd.DataFrame(rows, columns=["frame", "t", "x1", "y1", "x2", "y2", "ttc_s"])
    df.to_parquet(ride_dir / "video_unknown.parquet", index=False)
    print(f"{len(df)} unknown-object boxes in {df['frame'].nunique()} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    looming(args.ride_dir, args.video)
