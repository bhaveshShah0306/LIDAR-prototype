"""Bring a ride video into the project: 1080p H.264, no GPS data, one folder per ride.

Usage:
    uv run python pipeline/ingest.py "D:/phone/VID_20261029_083012.mp4" --weather sunny --note "office commute"

Creates data/rides/<date>_<n>/ with video.mp4 and meta.json.
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

RIDES = Path(__file__).resolve().parent.parent / "data" / "rides"
SHORT_SIDE = 1080  # 1920x1080 landscape, 1080x1920 portrait


def video_info(path):
    cap = cv2.VideoCapture(str(path))  # only reads the header, no frames are decoded
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not w:
        raise SystemExit(f"Can't read {path}")
    return w, h, fps, frames


def next_ride_dir(date):
    n = 1 + sum(1 for p in RIDES.glob(f"{date}_*") if p.is_dir())
    return RIDES / f"{date}_{n}"


def convert(src, dst, w, h):
    """Shrink the shorter side to 1080 (never enlarge), 30 fps, H.264, no audio, no metadata."""
    if min(w, h) > SHORT_SIDE:
        scale = f"scale=-2:{SHORT_SIDE}" if w > h else f"scale={SHORT_SIDE}:-2"
    else:
        scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # keep size; H.264 needs even numbers
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-stats",
           "-i", str(src), "-vf", f"{scale},format=yuv420p", "-r", "30",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2",
           "-an", "-map_metadata", "-1", str(dst)]
    subprocess.run(cmd, check=True)


def ingest(src, date=None, weather="", time_of_day="", note=""):
    src = Path(src)
    w, h, fps, frames = video_info(src)
    date = date or dt.date.today().isoformat()
    ride = next_ride_dir(date)
    ride.mkdir(parents=True)
    out = ride / "video.mp4"
    print(f"{src.name}: {w}x{h}, {fps:.0f} fps, {frames / fps:.0f} s -> {out}")
    convert(src, out, w, h)

    w2, h2, fps2, frames2 = video_info(out)
    meta = {
        "ride": ride.name, "date": date, "source": src.name,
        "width": w2, "height": h2, "fps": round(fps2, 2), "frames": frames2,
        "duration_s": round(frames2 / fps2, 1),
        "orientation": "landscape" if w2 > h2 else "portrait",
        "weather": weather, "time_of_day": time_of_day, "note": note,
        "split": None,  # set to "train" or "test" on Sat 31
    }
    (ride / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done: {ride}  ({w2}x{h2}, {meta['duration_s']} s)")
    if meta["orientation"] == "portrait":
        print("Note: portrait video. Mount the phone sideways for real rides.")
    return ride


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--date", help="YYYY-MM-DD, default today")
    ap.add_argument("--weather", default="")
    ap.add_argument("--time-of-day", default="", help="morning / afternoon / evening / night")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    ingest(args.video, args.date, args.weather, args.time_of_day, args.note)