# LIDAR-prototype: rear-camera traffic alert

A rear-facing phone camera on a Honda Activa 3G watches the traffic behind the scooter.
Python detects and tracks what's there (named classes from a fine-tuned YOLO, plus
unidentified objects found with optical flow), estimates time-to-collision, and a trained
risk model lights a left or right warning light on the handlebar.

Status: planning. See [PLAN.md](PLAN.md) for the day-by-day plan (28 Oct - 8 Nov 2026).


## Fair test: unseen ride 2026-11-20_1 (landscape, 341 s, split = test)
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