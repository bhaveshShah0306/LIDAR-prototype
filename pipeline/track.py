"""Give every object an ID that stays the same from frame to frame (ByteTrack).

Usage:
    uv run python pipeline/track.py data/rides/2026-10-28_1/video.mp4

Writes next to the video:
    <name>_tracks.parquet  frame, t, track_id, class, conf, x1, y1, x2, y2
    <name>_tracked.mp4     boxes labelled "#id class"
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from detect import CLASS_MAP, COLOURS, clean, draw

COLUMNS = ["frame", "t", "track_id", "class", "conf", "x1", "y1", "x2", "y2"]


def track(video, model_name="yolo11n.pt", fps=10.0, conf=0.4, imgsz=640, width=1280,
        save_video=True):
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Can't open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))

    model = YOLO(model_name)
    out_table = video.with_name(f"{video.stem}_tracks.parquet")
    out_video = video.with_name(f"{video.stem}_tracked.mp4")
    writer = None
    rows = []
    frame_idx = processed = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            if frame.shape[1] > width:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            t = frame_idx / src_fps

            # ByteTrack also uses weak boxes to keep IDs alive through blur and overlap,
            # so let it see everything above 0.1 and apply our own `conf` afterwards.
            result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.1,
                                 imgsz=imgsz, classes=list(CLASS_MAP), verbose=False)[0]
            boxes = result.boxes
            dets = []
            if boxes.id is not None:  # None until the tracker has confirmed something
                dets = [[CLASS_MAP[int(k)], c, *xyxy, tid] for xyxy, c, k, tid in
                        zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(),
                            boxes.id.int().tolist()) if c >= conf]

            # clean() returns the same lists it was given, so the track id at the end survives
            for name, c, x1, y1, x2, y2, tid in clean(dets, frame.shape[0]):
                rows.append([frame_idx, round(t, 3), tid, name, round(c, 3), x1, y1, x2, y2])
                if save_video:
                    draw(frame, x1, y1, x2, y2, f"#{tid} {name}", COLOURS[name])

            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps / step, (w, h))
                writer.write(frame)

            processed += 1
            if processed % 50 == 0:
                print(f"{frame_idx / max(total, 1):6.1%}  "
                      f"{processed / (time.time() - start):5.1f} frames/s")
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if processed == 0:
        raise SystemExit(f"No frames could be read from {video}")

    df = pd.DataFrame(rows, columns=COLUMNS)
    # TODO (Thu): drop tracks shorter than 0.5 s, they're flicker rather than real objects
    df.to_parquet(out_table, index=False)
    print(f"\nDone in {time.time() - start:.0f} s: {processed} frames, {len(df)} boxes, "
          f"{df['track_id'].nunique()} tracks")
    if len(df):
        print(df.groupby("class")["track_id"].nunique().rename("tracks").to_string())
    print(f"Table: {out_table}\nVideo: {out_video}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    track(args.video, args.model, args.fps, args.conf, width=args.width,
          save_video=not args.no_video)
