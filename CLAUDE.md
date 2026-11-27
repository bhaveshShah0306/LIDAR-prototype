# CLAUDE.md: working memory for Claude sessions on this repo

Claude Code reads this file at the start of every session. Keep it short and current:
the full project record (results, debug log, decisions) is NOTES.md.
Update the "Current state" section at the end of each working session.

## Who I am and how we work
- Bhavesh, full-stack developer (.NET 8, MS SQL Server, Angular), moving into AI/ML.
  Goal: an AI-centred portfolio project, and genuinely learning Python, computer vision and
  supervised learning.
- **Teaching mode:** I write the code. Give skeletons with `___` blanks and TODOs, hints first,
  full code only when I ask. Explain the "why", add "check yourself" questions, and give a small
  hand-checkable test (REPL lines or numbers to predict) for each function.
- Measured results over claims. Be honest about numbers and limits. Test a hypothesis with data
  before building a fix for it.
- Office Windows laptop, PowerShell in VS Code. Commands must be PowerShell.
  One command at a time for anything that deletes; full paths for deletes; never delete via a
  variable that may be empty.
- Run Python with `uv run python ...` from the project folder; add packages with `uv add`.
  In the REPL, `import sys; sys.path.insert(0, "pipeline")` first; after editing a module,
  `importlib.reload(module)`. Multi-line blocks need an empty line to finish.
- Office network inspects HTTPS: `$env:UV_SYSTEM_CERTS = "true"`; downloads in Python call
  `truststore.inject_into_ssl()` first.
- VS Code auto-imports sometimes add wrong lines (e.g. `from tracemalloc import start`): check imports.

## Git rules
- **Never change global git config on this PC** (global identity is the office one).
- Per-repo identity only: `user.name "Bhavesh Shah"`, `user.email shah.k.bhavesh@gmail.com`.
- **Don't push unless I ask.** Don't rewrite history or change commit authors.
- `data/`, `models/`, `*.pt`, `*.mp4`, `.venv/` are gitignored. Only public data; blur faces
  and number plates in anything published.
- The AI-Folds repo is parked: don't touch it.

## Project: Rear Alert (camera-only, AI-centred)
Rear-facing phone camera on a Honda Activa 3G -> detect and track traffic behind -> left/right
warning lights (drawn on screen; ESP32 LEDs optional).

**Purpose:** cover the zone behind the rider that he can't see; warn when a vehicle in that
frame comes close or closes in fast (centre vehicles included). Awareness aid, not full
collision avoidance.
**Scope:** one rear camera, only what is inside its frame. Out of scope: side/front coverage,
vehicles fully alongside (outside the frame), sudden swerves (mine or others'). No ultrasonic
sensors, no optical-flow/gyro "my own swerve" detection (future work at most).

## Pipeline (pipeline/)
`ingest.py` -> `track.py` (YOLO + ByteTrack, 10 fps, frames resized to max 1280 px wide; all
boxes in that size) -> `clean_tracks.py` -> `geometry.py` -> `rules.py` (baseline) and
`events.py` (labels) -> `features.py` -> `risk.py` (AI #2, LightGBM vs the rule).
Tools: `review.py`, `judge_alarms.py` (key-per-alarm verdicts into alarms_tally.csv),
`inspect_tracks.py`, `track_crops.py`, `debug_frame.py`, `calibrate.py`.
- Map classes BY NAME via `detect.class_map_for(model)`, never by number.
- Classes (id order fixed): 0 car, 1 bus_truck, 2 two_wheeler, 3 auto_rickshaw, 4 person,
  5 animal, 6 other.

## Finalized rules and definitions
- **Mirror:** image-left = rider's RIGHT, image-right = rider's LEFT. In videos, the R light is
  drawn top-left, the L light top-right (each above the part of the image it watches).
- **Zones:** thirds of the frame on box centre x (`SIDE_FRACTION = 1/3`). CENTRE counts for
  BOTH lights. Indian two-wheelers overtake on both sides: same thresholds both sides.
- **TTC** = 1 / slope of ln(box height), from `.shift(n)` over 0.5 s within a track (past only).
  Shrinking/steady/too-new -> inf; > 20 s -> inf. TTC is the main trigger (adapts to closing speed).
- **Distance** `dist_m = k / (y2 - y_horizon)` from the box BOTTOM (ground contact). Valid only
  below the horizon and up to 50 m. Needs `data/calibration.json` (calibrate.py, tape marks).
  Close vehicles cut off by the frame bottom: distance unreliable -> rely on TTC / edge events.
- **Corridor:** scooter half-width (`HALF_WIDTH_M = 0.45`, confirm with tape) + 1 m each side,
  sideways metres from ground contact (`add_lateral`). Defined in metres, not degrees.
- **Rule baseline (rules.py):** red TTC < 1.5 s, amber TTC < 2.5 s; distance red < 3 m,
  amber < 5 m ONLY while closing (finite TTC), so jams don't keep lights on.
  Then `fill_missing_frames` (rows = time) and `apply_hold` (0.5 s rolling max, backwards only).
- **Labels (events.py), near-miss / surrogate safety measures** (crashes can't be recorded):
  label 1 if within the next 2 s the track has
  A. closer than its class danger distance AND inside the corridor (needs calibration),
  B. passed alongside (last row big and at the left/right edge),
  C. too close to measure (big and at the bottom edge).
  Class danger distances (starting values from riding experience, not facts):
  two_wheeler/person/animal 2 m, car/auto/other 4 m, bus_truck 6 m.
  Labels look into the future; features must never (no leakage). Split train/val/test BY RIDE.
- **Judging guideline (for any by-eye verdict):** Q1 did the vehicle really get closer in the
  2 s before? no -> artefact. Q2 heading within ~1 m of my path? yes -> real, no -> wide.
  By-eye judgement is unreliable for subtle approaches: prefer objective labels.

## Key results so far
- Detector v2 (`models/fgvd_v2.pt`, yolo11n on FGVD + 206 own frames x5): bus_truck crop
  precision on the unseen test ride 2026-11-20_1: v1 ~10% -> v2 ~79%. ~14 fps on CPU.
- Rule baseline on 2026-11-20_1: 31 red alarms in 5.7 min. Round 2 (with guideline):
  real 10, artefact 15, wide 6 -> precision 32%.
- Analysis: artefacts are not young tracks, not edge boxes, not jitter (big boxes, steady
  1 s growth); only 3/31 were clear box jumps. Nearly every approaching vehicle slows or passes,
  so TTC alone over-warns. Class matters: 7/10 real alarms were two-wheelers; cars + autos gave
  10/15 artefacts. Conclusion: several weak signals -> learned model (AI #2), not more hand rules.

## Current state (update every session)
- Last commits: rules.py frame definitions; horizon and collision markers on the processed
  video; `add_lateral` in geometry.py; features on a fake ride.
- Written and working: geometry.py, rules.py, features.py.
- Design finalized but functions still TODO (`raise NotImplementedError`):
  events.py (`close_events`, `edge_events`, `label_track`), calibrate.py (`click_rows`, `fit`,
  `rms_error`), risk.py (all). looming.py is an optional extra, also TODO.
- **Next:**
  1. Write events.py `edge_events` + `label_track` (B and C need no calibration) and test on
     2026-11-20_1; `close_events` (A) can be written now and starts finding events after calibration.
  2. Write calibrate.py `fit` + `rms_error` (testable with made-up numbers) and `click_rows`.
     Field: measure camera height (tape, scooter upright); marks at 3, 5, 8, 12, 20 m;
     rms_error_m < 0.5 m. Then re-run geometry.py and events.py.
  3. Record 2-3 more landscape rides (same mount, stabilisation off, 1x lens).
  4. features.py on real rides -> risk.py: LightGBM vs rule on a held-out ride
     (precision, recall, false alarms/min, warning lead time).
- Open: check rider person boxes; 4 unchecked empty frames; `pipeline/judge_alarms.py` is not in
  the repo yet (re-create it if needed).
- Parked next project: Visual Quality-Inspection Copilot (separate repo, not started).
