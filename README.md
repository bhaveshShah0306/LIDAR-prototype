# LIDAR-prototype: rear-camera traffic alert

A rear-facing phone camera on a Honda Activa 3G watches the traffic behind the scooter.
Python detects and tracks what's there (named classes from a fine-tuned YOLO, plus
unidentified objects found with optical flow), estimates time-to-collision, and a trained
risk model lights a left or right warning light on the handlebar.

Status: planning. See [PLAN.md](PLAN.md) for the day-by-day plan (28 Oct - 8 Nov 2026).
