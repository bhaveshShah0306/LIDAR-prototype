"""MicroPython for the ESP32: drive two WS2812 LED sticks from lines sent by alert_out.py.

Copy to the board as main.py (Thonny: File -> Save as -> MicroPython device). It runs at power-on.

Wiring (ESP32 DevKit):
    left stick  DIN -> GPIO 5      right stick DIN -> GPIO 18
    both sticks 5V  -> VIN (USB 5 V),  GND -> GND
    (8 LEDs at full white can draw ~0.5 A per stick; keep brightness modest)

Protocol from the laptop, one line at a time:
    "L<0|1|2> R<0|1|2>"   0 off, 1 amber, 2 red flashing
    "H"                   heartbeat; no line for FAILSAFE_MS -> slow blue blink ("not protecting you")

Done when: typing "L1 R2" in Thonny's shell lights left amber and right flashing red,
and unplugging the laptop side makes both sticks blink blue within 2 seconds.

MicroPython you'll practise: pins, neopixel, non-blocking input with select.poll, time.ticks_ms.
"""
import select
import sys
import time

import machine
import neopixel

LEDS = 8
BRIGHT = 60  # 0..255
AMBER, RED, BLUE, OFF = (BRIGHT, BRIGHT // 3, 0), (BRIGHT, 0, 0), (0, 0, BRIGHT), (0, 0, 0)
FAILSAFE_MS = 2000
FLASH_MS = 125  # red flashes at 4 Hz

left = neopixel.NeoPixel(machine.Pin(5), LEDS)
right = neopixel.NeoPixel(machine.Pin(18), LEDS)
poll = select.poll()
poll.register(sys.stdin, select.POLLIN)


def fill(stick, colour):
    """Set every LED on one stick to `colour` and send it to the strip.

    TODO: for i in range(LEDS): stick[i] = colour   then   stick.write()
    """
    raise NotImplementedError("TODO: fill")


def parse(line, current):
    """Return the new (left_level, right_level) from a line like "L1 R2", or `current` if invalid.

    TODO: split on spaces; for each part, "L" + digit -> left, "R" + digit -> right.
          Wrap int() in try/except ValueError: never let a garbled line crash the firmware.
    """
    raise NotImplementedError("TODO: parse")


def colour_for(level, now):
    """OFF, AMBER, or RED/OFF alternating every FLASH_MS for level 2.

    TODO: for level 2 use (now // FLASH_MS) % 2 to switch between RED and OFF.
    """
    raise NotImplementedError("TODO: colour_for")


levels = (0, 0)
last_msg = time.ticks_ms()
while True:
    now = time.ticks_ms()
    # TODO:
    #   1. while poll.poll(0): line = sys.stdin.readline().strip(); last_msg = now;
    #      if line != "H": levels = parse(line, levels)
    #   2. if time.ticks_diff(now, last_msg) > FAILSAFE_MS: blink BLUE on both (every 500 ms)
    #      else: fill(left, colour_for(levels[0], now)); fill(right, colour_for(levels[1], now))
    #   3. time.sleep_ms(20)
    raise NotImplementedError("TODO: main loop")
