"""Find vehicles, people and animals in a ride video with YOLO.

Usage:
    uv run python pipeline/detect.py data/rides/ride1.mp4

Writes next to the video:
    <name>_detections.parquet  one row per box: frame, t, class, conf, x1, y1, x2, y2
    <name>_annotated.mp4       the sampled frames with the boxes drawn on them
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

try:  # office network: trust the Windows certificate store when downloading the model
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Pretrained COCO class id -> our class. Anything not listed is ignored.
CLASS_MAP = {
    0: "person",
    1: "two_wheeler",   # bicycle
    3: "two_wheeler",   # motorcycle
    2: "car",
    5: "bus_truck",     # bus
    7: "bus_truck",     # truck
    16: "animal",       # dog
    17: "animal",       # horse
    18: "animal",       # sheep
    19: "animal",       # cow
}
COLOURS = {  # BGR, as OpenCV expects
    "person": (0, 200, 255),
    "two_wheeler": (255, 160, 0),
    "car": (0, 220, 0),
    "bus_truck": (200, 0, 200),
    "animal": (0, 120, 255),
}
COLUMNS = ["frame", "t", "class", "conf", "x1", "y1", "x2", "y2"]


def detect(video, model_name="yolo11n.pt", fps=10.0, conf=0.3, imgsz=640, save_video=True):
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Can't open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))  # 30 fps video, 10 fps wanted -> every 3rd frame

    model = YOLO(model_name)  # downloads yolo11n.pt (~6 MB) the first time
    out_table = video.with_name(f"{video.stem}_detections.parquet")
    out_video = video.with_name(f"{video.stem}_annotated.mp4")
    writer = None
    rows = []
    frame_idx = processed = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            t = frame_idx / src_fps
            result = model.predict(frame, conf=conf, imgsz=imgsz,
                                   classes=list(CLASS_MAP), verbose=False)[0]
            boxes = result.boxes
            for (x1, y1, x2, y2), c, k in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                              boxes.cls.tolist()):
                name = CLASS_MAP[int(k)]
                rows.append([frame_idx, round(t, 3), name, round(c, 3), x1, y1, x2, y2])
                if save_video:
                    colour = COLOURS[name]
                    thick = max(2, frame.shape[1] // 400)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, max(1, int(thick * 0.6)))
                    cv2.putText(frame, f"{name} {c:.2f}", (int(x1), max(20, int(y1) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, thick * 0.4, colour, thick)
                    cv2.putText(frame, f"{name} {c:.2f}", (int(x1), max(15, int(y1) - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, thick * 0.6, colour, thick)

            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             src_fps / step, (w, h))
                cv2.putText(frame, f"t={t:6.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)
                writer.write(frame)

            processed += 1
            if processed % 50 == 0:
                speed = processed / (time.time() - start)
                left = (total - frame_idx) / step / speed if total else 0
                print(f"{frame_idx / max(total, 1):6.1%}  {speed:5.1f} frames/s  "
                      f"~{left / 60:.1f} min left")
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_parquet(out_table, index=False)
    print(f"\nDone in {time.time() - start:.0f} s: {processed} frames, {len(df)} boxes")
    print(df["class"].value_counts().to_string() if len(df) else "No detections")
    print(f"Table: {out_table}")
    if save_video:
        print(f"Video: {out_video}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second to analyse")
    ap.add_argument("--conf", type=float, default=0.3, help="minimum confidence to keep a box")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    args = ap.parse_args()
    detect(args.video, args.model, args.fps, args.conf, args.imgsz, not args.no_video)