# Rear Alert: project notes

Rear-facing camera on a Honda Activa 3G -> detect and track traffic behind the scooter ->
warn the rider (left/right lights) when something approaches dangerously.
Goal: an AI-centred learning project (computer vision + supervised learning), built in Python.

---


**Scope:** one rear-facing phone camera. The system only sees, and warns about, what is inside
that camera's frame: no side or front coverage, no blind-spot handling. Distance is valid only
where a vehicle's contact with the road is visible in the frame; very close vehicles cut off by
the frame edge are handled by time-to-collision (TTC), which needs no distance.

## 1. Setup and environment

- Laptop: Dell Latitude 3420, i5-1135G7 (4 cores, Iris Xe), 16 GB RAM, no NVIDIA GPU.
  Training runs on Kaggle (free T4 GPU); everything else runs locally.
- Python 3.10 project managed with `uv` (`pyproject.toml`, `uv.lock`, `.python-version`).
  Ritual: Init -> Pin -> Venv -> Add -> Run -> Commit. Daily: `git pull` -> `uv sync` -> `uv run ...`.
- Office network inspects HTTPS:
  - uv downloads: `$env:UV_SYSTEM_CERTS = "true"`
  - Python downloads (YOLO weights): `truststore.inject_into_ssl()` at the top of scripts
- OpenCV pinned to `opencv-python<5` (5.0 pre-release ran out of memory on 4K frames).
- ffmpeg comes from the `imageio-ffmpeg` package (no system install needed).
- Memory: office apps can leave < 1 GB free. Close Chrome/Teams, or restart, before heavy runs.
- Git: this repo uses my personal identity, set per repo only (no global changes):
  `git config user.name "Bhavesh Shah"`, `git config user.email "shah.k.bhavesh@gmail.com"`,
  remote URL includes the username `bhaveshShah0306@github.com/...`.
  Early commits were made with the office email (global default) before this was set.
- Labelling tool: X-AnyLabeling v3.3.10, installed separately with `uv tool install`.

## 2. Recording rules (learned the hard way)

- Phone mounted on the rear grab rail, facing backwards, LANDSCAPE, 1080p (or 720p), 30 fps.
- Same mount position every ride (tape mark).
- Free storage before riding; play the full video on the phone before copying.
- Copy the ORIGINAL file (USB / original-quality upload). Check the size: ~4-5 min of
  video should be well over 100 MB. A file far smaller is truncated or recompressed.
- New videos go into `data/raw/`; only `ingest.py` creates folders in `data/rides/`.

## 3. Data

| Ride | Source | Format | Use |
|---|---|---|---|
| 2026-10-28_1 | first test clip | 4K portrait, 13 s | code testing only |
| 2026-11-03_1 | colleague's ride | 720x1280 portrait HEVC; damaged, only the first 70 s of 229 s survive | hard-case test, not training |
| 2026-11-12_1 | my ride | 1280x720 landscape, 310 s | **train** (207 labelled frames) |

FGVD (Fine-Grained Vehicle Detection, IIIT Hyderabad / IDD site), non-commercial licence:
- 5,502 images, 215 labels (type_make_model), 24,450 boxes, Pascal VOC XML.
- 7 vehicle types: car 7,951, motorcycle 5,293, scooter 4,347, autorickshaw 3,778,
  truck 1,552, bus 1,215, mini-bus 315. Splits: train 3,535 / val 884 / test.
- Filmed FORWARDS from a car: vehicles mostly seen from behind.

## 4. Classes (order is fixed: line number = class id)

`0 car, 1 bus_truck, 2 two_wheeler, 3 auto_rickshaw, 4 person, 5 animal, 6 other`

- COCO -> ours: person->person, bicycle/motorcycle->two_wheeler, car->car,
  bus/truck->bus_truck, dog/horse/sheep/cow->animal.
- FGVD -> ours: car->car, motorcycle/scooter->two_wheeler, autorickshaw->auto_rickshaw,
  truck/bus/mini-bus->bus_truck.
- Always map classes BY NAME (`class_map_for(model)`), never by number: every model numbers
  its classes differently (COCO 3 = motorcycle, ours 3 = auto_rickshaw).

## 5. Pipeline (pipeline/)

| Script | Does |
|---|---|
| ingest.py | raw video -> 1080p H.264, GPS metadata stripped, `meta.json` per ride |
| detect.py | YOLO per frame at 10 fps, resize to <= 1280 wide; `clean()` drops boxes < 6% frame height and merges riders into their bike |
| track.py | ByteTrack IDs; tracker sees boxes >= 0.1 conf, we keep >= 0.4 |
| clean_tracks.py | drop tracks < 0.5 s, majority-vote class per track, rolling-median box smoothing |
| review.py | step through a ride, sliders for conf / min height, mark mistakes to `review_marks.csv` |
| explore.py | DuckDB SQL over the Parquet files |
| debug_frame.py | one frame, every raw box, two models side by side, optional 16:9 crop |
| inspect_tracks.py | per-track summary + suspicious flags (too wide, both edges, flicker, low conf, class switch) |
| track_crops.py | one crop per track of a class, for judging by eye |
| list_labels.py / convert_fgvd.py | explore FGVD labels; VOC XML -> YOLO, plus pseudo-labelled people/animals |
| check_labels.py | draw YOLO labels back on images to verify a dataset |
| extract_frames.py | sample frames + pre-label them for labelling |
| yolo_to_labelme.py / labelme_to_yolo.py | YOLO <-> X-AnyLabeling JSON |

Speed on the laptop: detection ~3.7 fps on 4K input (decode-bound); tracking ~10 fps at 720p;
YOLO11n inference ~34 ms per image on CPU.

## 6. AI #1: detector

### Why fine-tune
- Pretrained YOLO (COCO, 80 classes) has no auto-rickshaw class. It must pick one of its
  classes (closed set), so autos came out as "truck" -> bus_truck.

### Dataset build
- FGVD VOC -> YOLO with our 7 classes (`convert_fgvd.py`).
- FGVD labels only vehicles, so people/animals were PSEUDO-LABELLED with yolo11n (conf >= 0.5);
  otherwise the model would learn "people are background".
- Verified visually with `check_labels.py` (first version had a bug: forgot to multiply the
  normalised values by image size, all boxes collapsed to (0, 0)).

### Training v1 (Kaggle, T4)
- yolo11n.pt (transfer learning: 499/499 weights transferred, head rebuilt for 7 classes),
  50 epochs, imgsz 640, batch 32, patience 15, ~43 min.
- Copied the dataset from /kaggle/input (network drive, 1.7 MB/s) to /kaggle/temp (3.7 GB/s).
- Model: `models/fgvd_v1.pt` (5.4 MB).

### Results on FGVD validation (884 images)
| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.875 | 0.685 | 0.755 | 0.618 |
| car | 0.871 | 0.843 | 0.924 | 0.791 |
| bus_truck | 0.819 | 0.786 | 0.853 | 0.716 |
| two_wheeler | 0.838 | 0.811 | 0.882 | 0.623 |
| auto_rickshaw | 0.886 | 0.826 | 0.914 | 0.797 |
| person | 0.835 | 0.847 | 0.917 | 0.757 |
| animal | 1.0 | 0 | 0.038 | 0.023 |

- Before fine-tuning: auto_rickshaw = no such class.
- "all" is dragged down by `animal` (1 validation example). Without it mAP50 ~ 0.90.
- Person numbers are optimistic: they measure agreement with pseudo-labels.
- mAP50-95 was still rising at epoch 50.

## 7. Debug log

### 7.1 Giant full-width bus_truck box (portrait ride 2026-11-03)
- Observed: track #109, box 99-100% wide, 72% tall, frames 261 to 312+, conf 0.40-0.80.
- H1 low confidence: REJECTED (conf up to 0.80).
- H2 portrait input: CONFIRMED (gone in a 16:9 crop).
- H3 learned from FGVD: CONFIRMED (yolo11n does not produce it).
- On the landscape ride 2026-11-12_1: 0 boxes wider than 80% -> fixed by recording landscape.

### 7.2 Pedestrians
- Portrait frame 273: yolo11n found 14 people, fgvd_v1 found 0.
- Landscape ride: fgvd_v1 tracked 250 person tracks -> mostly a portrait problem too.

### 7.3 Close vehicles labelled bus_truck (landscape ride)
- 242 bus_truck tracks: 177 flickers (73%, 1-4 frames), 55 of 5-19 frames, 10 of 20+.
- 65 tracks of 5+ frames judged by eye from crops:
  real bus/truck 3, auto 31, car 26, other 4, unsure 1.
- **bus_truck precision on my footage: 3/64 = 4.7% vs 0.82 on FGVD validation.**
- Pattern: CLOSE vehicles (box 44-85% of frame height), confident (0.78-0.89),
  often cut off at the frame edge.
- Cause: VIEWPOINT. FGVD's forward camera shows vehicles from behind; my rear camera sees
  them from the front and close up. Not fixable by thresholds: needs my own training data.

### 7.4 Track cleaning (clean_tracks.py on 2026-11-12_1)
| Class | Tracks raw -> clean | Boxes raw -> clean |
|---|---|---|
| auto_rickshaw | 193 -> 84 | 1,363 -> 1,373 |
| bus_truck | 242 -> 61 | 1,205 -> 829 |
| car | 332 -> 152 | 2,293 -> 2,148 |
| person | 250 -> 83 | 1,680 -> 1,482 |
| two_wheeler | 261 -> 79 | 1,148 -> 887 |
| total | 1,278 -> 459 (-64%) | 7,689 -> 6,719 (-13%) |
- Mostly flickers removed; the majority vote moved short mislabels back to auto/car.
- 61 bus_truck tracks stay wrong as whole tracks -> retraining, not cleaning.

## 8. Own labels, round 1 (2026-11-12_1, 207 frames, every 1.5 s)

- Pre-labels came from the COCO model (yolo11n) because extract_frames.py still had the old
  default. Next time pre-label with fgvd_v1.
- Labelled in X-AnyLabeling, exported with my own `labelme_to_yolo.py`
  (the app's export did not write the files).
- Backup of the pre-labels: `data/labels/own/train/labels_prelabel/`.

| Class | Pre-labels | My labels |
|---|---|---|
| car | 459 | 464 |
| bus_truck | 216 | 34 (-84%) |
| two_wheeler | 242 | 358 |
| auto_rickshaw | 0 | 279 |
| person | 462 | 338 |
| other | 0 | 16 |
| total | 1,379 | 1,489 |

- TODO: check riders consistently have a person box (person count dropped by 124).
## 9. Detector v2 (`models/fgvd_v2.pt`)

### Training
- Start: yolo11n.pt (COCO), NOT continued from v1 -> one clean recipe, fair comparison.
- Data: FGVD train (3,535) + my 206 rear-facing frames listed 5x (oversampling)
  -> ~4,565 training images per epoch (143 batches of 32). Validation: FGVD val (884).
- Same settings as v1: 50 epochs, imgsz 640, batch 32, patience 15, Kaggle T4 (~40 min).
- Kaggle lessons from this round:
  - The first v2 run was interactive: its /kaggle/working output (incl. best.pt) was deleted
    when the session ended. Re-ran as "Save & Run All" (committed version), where outputs persist.
  - A half-finished dataset copy in /kaggle/temp made the old "folder exists?" check skip the
    copy -> FileNotFoundError. Fixed by comparing file counts and re-copying if incomplete.
  - Several runs at once each use GPU quota: check Active Events, stop idle sessions.
  - The downloaded .pt arrived as .zip: a .pt IS a zip archive inside; just rename it.
  - Verified with ckpt['train_args']['name'] == 'v2'.

### FGVD validation (does v2 still know FGVD?)
| Class | v1 mAP50 | v2 mAP50 | v1 mAP50-95 | v2 mAP50-95 |
|---|---|---|---|---|
| car | 0.924 | 0.919 | 0.791 | 0.782 |
| bus_truck | 0.853 | 0.835 | 0.716 | 0.697 |
| two_wheeler | 0.882 | TODO | 0.623 | TODO |
| auto_rickshaw | 0.914 | 0.904 | 0.797 | 0.783 |
| person | 0.917 | 0.911 | 0.757 | 0.742 |
| animal | 0.038 | 0.332 | 0.023 | 0.133 |
| all | 0.755 | 0.796 | 0.618 | 0.623 |
- No meaningful forgetting: real classes changed by -0.5 to -2 points (normal cost of learning
  a second viewpoint).
- The higher "all" comes from `animal` (1 validation example) -> noise, not an improvement.
- Table is from the first (interactive) v2 run; the kept model is the committed re-run
  (deterministic, seed 0, so numbers should match closely). TODO: copy that run's table.

### On my training ride 2026-11-12_1 (sanity check - NOT a fair test, v2 saw 206 of its frames)
Clean tracks (clean_tracks.py):
| Class | v1 tracks | v2 tracks | v1 boxes | v2 boxes |
|---|---|---|---|---|
| bus_truck | 61 | 18 | 829 | 292 |
| auto_rickshaw | 84 | 206 | 1,373 | 3,555 |
| car | 152 | 265 | 2,148 | 4,251 |
| two_wheeler | 79 | 191 | 887 | 3,633 |
| person | 83 | 112 | 1,482 | 1,159 |
| other | 0 | 12 | 0 | 120 |
| total | 459 | 804 | 6,719 | 13,010 |
- v2 finds almost twice as many objects: far more confident on the rear view.
- bus_truck -70%, auto_rickshaw x2.5, `other` now detected.
- Person boxes down while person tracks up -> check riders still get a person box
  (my labels had fewer person boxes than the pre-labels).

bus_truck tracks of 5+ frames, judged by eye from crops (track_crops.py):
| Model | tracks | real bus/truck | auto | car | other | precision |
|---|---|---|---|---|---|---|
| v1 | 65 | 3 | 31 | 26 | 4 | 4.7% |
| v2 | 18 | 16 | 2 | 0 | 0 | 89% |
("real" includes goods three-wheelers / tempos, per the labelling rules.)

Example, t = 117 s (frame 3510):
- close white car cut off at the edge: v1 bus_truck -> v2 car 0.96
- close auto: v1 bus_truck -> v2 auto_rickshaw
- second car behind: v1 missed -> v2 car 0.91

Speed: v2 + ByteTrack at 1280x720: 3,103 frames in 216 s = ~14.4 fps on the i5 CPU (no GPU)
-> faster than the 10 fps the pipeline needs.

### Open points
- 2 autos still labelled bus_truck: look at what they have in common.
- Small pedestrians: check whether v2 boxes them (lower min-height slider in review.py).
- FAIR TEST: new landscape ride (unseen, ideally another route) -> run v1 and v2 -> same crop
  check -> optionally label ~100 test frames and compare mAP on my own footage.

### Labelling rules (apply the same way every time)
- One box per vehicle, never one box over two vehicles. Boxes may overlap.
- Partly hidden: box the visible part if about a third or more is visible; skip if > ~70% hidden.
- Cut off by the frame edge: box the visible part, with the correct class.
- Passenger auto -> auto_rickshaw. Goods three-wheeler, tempo, mini-truck, bus, truck -> bus_truck.
- E-rickshaw, handcart, tractor, cycle-rickshaw, anything odd -> other.
- Motorbike/scooter -> two_wheeler. Rider -> ALSO a person box (matches FGVD pseudo-labels).
- Every visible pedestrian -> person, even small ones. Animals -> animal.
- Parked vehicles: label them.
- Skip objects under ~10 px tall.
- Boxes on nothing (shop fronts, shadows) -> delete (hard negatives).
- Frames that are not road views (camera covered at start/end) -> leave out of training.
- A frame is only used for training when it is FULLY labelled.

## 9. Concepts learned

- Transfer learning / fine-tuning; closed-set classification; pseudo-labelling.
- Precision, recall, mAP50, mAP50-95; per-class vs averaged metrics.
- Domain gap (viewpoint, orientation) measured on my own data.
- Hard negatives; data-centric AI / active-learning loop: review -> label -> retrain -> measure.
- Tracking: ByteTrack, flickers, ID switches; cleaning with groupby/transform.
- Debugging process: observe -> measure -> hypotheses -> test one change -> record.

## 10. Process lessons

- Notebook cells run in the order executed; an error stops the cell; a restart clears variables
  and /kaggle/working. Prefer one self-contained cell. Use "Save & Run All" for long runs.
- Kaggle input folders are slow network drives: copy to local disk before training.
- Laptop sleep / dropped browser connection freezes the notebook view but the session keeps
  training. Download best.pt during training as a backup.
- Scripts must report what ACTUALLY happened (e.g. count successful writes), not what was intended.
- Never run delete commands with a variable that might be empty; use full paths.
- Holding Ctrl while dragging in Explorer COPIES files.
- Re-running a script that writes files can duplicate earlier output.
- PowerShell: one command at a time for anything that deletes; `"${d}:"` not `"$d:"` in strings.
- VS Code auto-imports (e.g. `from unicodedata import name`) can silently break code.

## 11. Decisions

- Camera-only + AI. No ultrasonic sensor: 0.25-4.5 m straight back, misses the side lanes.
  Closing the gap to radar is done with AI (own data, retraining, optional monocular depth).
- Handlebar lights shown on screen; ESP32 LEDs optional.
- Live demo later: phone stream -> laptop, tested standing still; never ride with the laptop.
- Zones: frame split into thirds on box centre x; image is mirrored (image-left = rider's RIGHT).
  Indian two-wheelers overtake on BOTH sides, so both lights use the same thresholds, and CENTRE
  boxes count for both sides.
- TTC (from box-height growth over 0.5 s) is the main trigger: it adapts to closing speed
  (warns at ~4.5 m for a 3 m/s overtake, ~12 m for an 8 m/s bike).
- Danger zone from riding experience: within 3-5 m. Distance rule (red < 3 m, amber < 5 m) only
  counts while the vehicle is closing (finite TTC), so a jam doesn't keep the light on.
- Distance from ground contact: dist_m = k / (y2 - y_horizon), kept only up to 50 m
  (near the horizon 1 px ~ several metres). Needs calibration (calibrate.py, tape marks).
  - Considered and dropped for now: red light on MY sideways movement (optical flow / phone gyro),
  side ultrasonic sensors (future work: the right tool for vehicles fully alongside).
- Collision corridor (scooter + 1 m each side, sideways metres from ground contact):
  optional; build only if the alarm tally shows many "wide pass" false alarms.

## 12. Next steps

1. Check rider person boxes; re-run labelme_to_yolo.py if anything changes.
2. Test ride: new landscape ride -> ingest -> split "test" -> extract with fgvd_v1 (--every 3)
   -> label fully.
3. Train v2 on Kaggle: FGVD + own frames (own frames oversampled ~5x), evaluate on my test frames.
4. Compare v1 vs v2: mAP on my test frames, per-class recall, bus_truck precision from crops.
5. Then geometry.py (TTC, zones, distance), rules.py, events/features/risk.py (AI #2).
1. Alarm tally on 2026-11-20_1 (31 red alarms): real / false artefact / false wide pass /
   miss close-no-light / miss detector. This is the rule baseline AI #2 must beat.
2. Calibration: measure camera height (tape); field session with marks at 3, 5, 8, 12, 20 m.
3. Record 2-3 more landscape rides (same mount, stabilisation off) for AI #2.
4. AI #2: events.py (label: within 3 m while closing in the next 2 s), features.py, risk.py.
5. Open: check rider person boxes; the 4 unchecked empty frames.


## 14. Fair test: unseen ride 2026-11-20_1 (landscape, 341 s, split = test)
Clean tracks: bus_truck 91 -> 10, auto_rickshaw 56 -> 102, car 73 -> 115,
two_wheeler 100 -> 154, person 55 -> 99, other 0 -> 17 (v1 -> v2).

bus_truck tracks (5+ frames), judged blind from crops:
| Model | tracks | real | car | auto | other | precision |
|---|---|---|---|---|---|---|
| v1 | 93 | 9 | 26 | 45 | 1 | ~11% (9/81, 12 crops still to sort) |
| v2 | 14 | 11 | 0 | 2 | 1 | ~79% (11/14) |
- Close cars as bus_truck: 26 -> 0. Autos as bus_truck: 45 -> 2.
- Generalises to an unseen ride; lower than on the training ride (89%), as expected.
- Small sample (14); precision only, recall not yet measured.