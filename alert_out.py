"""Replay a ride's alerts on the real handlebar LEDs (laptop -> USB -> ESP32 -> LEDs).

Protocol (one text line per CHANGE, 115200 baud):
    "L<left> R<right>\\n"   left/right: 0 off, 1 amber, 2 red flashing      e.g.  "L0 R2\\n"
The ESP32 (firmware/main.py) also expects a heartbeat line "H\\n" at least every second;
without it, it blinks blue so the rider knows the system is NOT protecting them.

Usage:
    uv add pyserial
    uv run python alert_out.py data/rides/2026-11-03_2 --fake          # print instead of sending
    uv run python alert_out.py data/rides/2026-11-03_2 --port COM5     # real ESP32 (see Device Manager)

Done when: the LEDs follow the ride in real time, and you've measured the delay (target < 100 ms).

Python you'll practise: serial ports, timing loops, simple text protocols.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

BAUD = 115200
HEARTBEAT_S = 0.5


class FakePort:
    """Stands in for serial.Serial when there's no ESP32: prints what would be sent."""

    def write(self, data: bytes):
        print(data.decode().strip())

    def close(self):
        pass


def open_port(port):
    """A real serial port, or FakePort if port is None.

    TODO: import serial; return serial.Serial(port, BAUD, timeout=0)
          Wait ~2 s after opening: many ESP32 boards reset when the port opens.
    """
    if port is None:
        return FakePort()
    raise NotImplementedError("TODO: open_port")


def play(levels: pd.DataFrame, port):
    """Send a line whenever left/right changes, at real-time speed, plus heartbeats.

    TODO:
      1. start = time.perf_counter(); last = None; last_beat = 0
      2. For each row (t, left_level, right_level), in order:
           - wait until perf_counter() - start >= t (sleep in small steps, sending "H\\n"
             every HEARTBEAT_S while you wait)
           - if (left, right) != last: port.write(f"L{left} R{right}\\n".encode()); last = (left, right)
      3. At the end send "L0 R0\\n".

    Check yourself: why send only CHANGES instead of a line for every frame?
    """
    raise NotImplementedError("TODO: play")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ride_dir")
    ap.add_argument("--port", help="e.g. COM5; leave out with --fake")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    ride = Path(args.ride_dir)
    src = ride / "video_risk.parquet"
    if not src.exists():  # the rule baseline works too, until risk.py is done
        src = ride / "video_rules.parquet"
    port = open_port(None if args.fake else args.port)
    try:
        play(pd.read_parquet(src), port)
    finally:
        port.close()
