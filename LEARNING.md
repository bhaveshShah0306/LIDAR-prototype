# Learning guide: build the rest yourself

The files already working (`ingest.py`, `detect.py`, `track.py`) were written for you. Everything
below is a **skeleton**: the plumbing (loading, saving, command-line arguments) is done, and every
function you need to write raises `NotImplementedError` with a `TODO` that explains what to do,
step by step. Running a skeleton tells you exactly which function is still missing.

## The routine for every file

1. **Read the module docstring** at the top: what goes in, what comes out, where it sits in the pipeline.
2. **Read every `TODO`** before writing anything. Sketch the steps on paper.
3. **Write one function at a time** and test it on a single ride. Print `df.head()` and `df.describe()` often.
4. **Answer the "Check yourself" questions** in the file, in your own words, as comments.
5. **Commit** when the "Done when" at the top is true.

Stuck for more than 20 minutes? Ask for help with: the full error, the function you're writing, and what you tried.
You'll get a hint first, and the full answer only if you ask for it.

## Order of work

| # | File | Plan day | What it does | Python / ML you'll learn |
|---|---|---|---|---|
| 0 | `pipeline/clean_tracks.py` (skeleton given in chat) | Day 2 | Drop flickers, one class per track, smooth boxes | pandas `groupby`, `transform`, `rolling` |
| 1 | `pipeline/calibrate.py` | Day 4 | Fit "pixel row -> metres" from tape marks | OpenCV mouse clicks, numpy `polyfit`, linearising a formula |
| 2 | `pipeline/geometry.py` | Day 3-4 | Zone, distance, time-to-collision per box | Pure functions, vectorised maths, rolling regressions |
| 3 | `pipeline/rules.py` | Day 3 | Baseline alert: hand-written TTC rule + overlay video | Time-series logic, hysteresis, drawing with OpenCV |
| 4 | `pipeline/extract_frames.py` | Day 4 | Pick frames to label + pre-label them | Sampling, file formats (YOLO labels) |
| 5 | `train/train_detector.py` | Day 5 | Fine-tune YOLO on Kaggle, compare before/after | **Supervised learning, transfer learning, mAP** |
| 6 | `pipeline/looming.py` | Day 6 | Unidentified objects from optical flow | Image processing, homography, gradients |
| 7 | `pipeline/events.py` | Day 6 | Automatic labels: "close pass within 2 s?" | **Label design, avoiding leakage** |
| 8 | `pipeline/features.py` | Day 6 | Features the model may use at time t | **Feature engineering, past-only windows** |
| 9 | `pipeline/risk.py` | Day 6 | Train + evaluate the alert model vs the rule | **LightGBM, train/val/test, precision/recall, thresholds** |
| 10 | `api/api.py` | Day 7 | FastAPI: rides, frames, events, WebSocket replay | async Python, REST, WebSockets |
| 11 | `web/` (React + three.js) | Day 8-9 | Dashboard | Your existing front-end skills |
| 12 | `alert_out.py` + `firmware/main.py` | Day 10 | Laptop -> ESP32 -> handlebar LEDs | Serial protocols, MicroPython |

## File contract (what each step reads and writes)

Every ride lives in `data/rides/<ride>/`:

| File | Written by | Columns |
|---|---|---|
| `video.mp4`, `meta.json` | `ingest.py` | |
| `video_tracks.parquet` | `track.py` | frame, t, track_id, class, conf, x1, y1, x2, y2 |
| `video_tracks_clean.parquet` | `clean_tracks.py` | same, cleaned |
| `video_geometry.parquet` | `geometry.py` | + cx, h, zone, dist_m, ttc_s |
| `video_rules.parquet` | `rules.py` | frame, t, left_level, right_level |
| `video_unknown.parquet` | `looming.py` | frame, t, x1, y1, x2, y2, ttc_s (class `unknown`) |
| `video_events.parquet` | `events.py` | geometry rows + label (0/1) |
| `video_features.parquet` | `features.py` | track_id, frame, t, feature columns, label |
| `video_risk.parquet` | `risk.py` | frame, t, left_level, right_level (model) |

Box coordinates are always in the **processed frame**: the video resized to at most 1280 px wide,
exactly as `track.py` does. Calibration must use the same size.

## Packages to add when you reach each file

```powershell
uv add duckdb                                   # explore / api
uv add scikit-learn lightgbm matplotlib         # risk.py
uv add fastapi "uvicorn[standard]"              # api.py
uv add --dev pytest httpx websockets            # api tests
uv add pyserial                                 # alert_out.py
```

## One thing that trips everyone up: left and right

The camera faces **backwards**. Something on the **left of the image** is on the rider's **right**.
India drives on the left, so overtakes happen on the rider's right, which is the **left of the image**.
`geometry.py` asks you to handle this; get it wrong and the wrong handlebar light comes on.
