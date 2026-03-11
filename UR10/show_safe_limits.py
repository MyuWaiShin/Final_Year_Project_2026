"""
show_safe_limits.py
====================
Live terminal viewer — loads Perception/safe_limits.json and streams the
current robot TCP position against the recorded workspace limits.

Colour coding (ANSI terminal):
  🟢 Green  = inside safe zone
  🟡 Yellow = within WARNING_MM of a limit
  🔴 Red    = outside the limit (VIOLATION)

Usage:
    python UR10/show_safe_limits.py

Run this in a second terminal while jogging the robot with the pendant.
Press Ctrl-C to quit.
"""

import json
import socket
import struct
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_IP      = "192.168.8.102"
LIMITS_PATH   = Path(__file__).parent.parent / "Perception" / "safe_limits.json"
WARNING_MM    = 20.0   # warn when within this many mm of a limit
POLL_HZ       = 10     # how often to poll robot pose

# ANSI colour codes
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CLEAR  = "\033[2J\033[H"   # clear screen + move cursor home


# ─────────────────────────────────────────────────────────────────────────────
# Robot pose reader
# ─────────────────────────────────────────────────────────────────────────────
def get_tcp_pose(ip=ROBOT_IP):
    """Returns [X, Y, Z, Rx, Ry, Rz] in metres/radians, or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))
            data = s.recv(1200)
            if len(data) < 492:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Limit check helpers
# ─────────────────────────────────────────────────────────────────────────────
def axis_status(value_m, lo_m, hi_m, warn_m):
    """
    Returns (colour, status_str, distance_to_nearest_mm).
    lo and hi are the SAFE limits (value must be between them).
    For X: lo = left limit (more negative), hi = right limit (more positive)
    """
    v = value_m
    mm = v * 1000
    lo_mm = lo_m * 1000
    hi_mm = hi_m * 1000

    dist_lo = v - lo_m   # positive = inside (above lo)
    dist_hi = hi_m - v   # positive = inside (below hi)

    # Determine which direction is more dangerous
    dist_nearest = min(abs(dist_lo), abs(dist_hi)) * 1000  # mm

    if v < lo_m or v > hi_m:
        return RED, "VIOLATION", dist_nearest
    elif dist_nearest < warn_m:
        return YELLOW, "WARNING ", dist_nearest
    else:
        return GREEN, "OK      ", dist_nearest


def fmt_axis(name, value_m, lo_m, hi_m, warn_m):
    colour, status, dist_mm = axis_status(value_m, lo_m, hi_m, warn_m)
    v_mm  = value_m * 1000
    lo_mm = lo_m * 1000
    hi_mm = hi_m * 1000
    bar   = _mini_bar(value_m, lo_m, hi_m, width=30)
    return (
        f"  {colour}{BOLD}{name}{RESET}  "
        f"current = {v_mm:+8.1f} mm   "
        f"limits [{lo_mm:+7.1f} .. {hi_mm:+7.1f}]   "
        f"{colour}{status}{RESET}  "
        f"(nearest edge: {dist_mm:.1f} mm)   {bar}"
    )


def _mini_bar(value, lo, hi, width=30):
    """ASCII progress bar showing position within limits."""
    if hi <= lo:
        return ""
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    pos  = int(frac * width)
    bar  = "[" + "─" * pos + "●" + "─" * (width - pos) + "]"
    return bar


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not LIMITS_PATH.exists():
        print(f"ERROR: {LIMITS_PATH} not found.")
        print("Run:  python Perception/record_safe_limits.py  first.")
        return

    with open(LIMITS_PATH) as f:
        data = json.load(f)

    limits = data["safe_limits"]
    SAFE_X = limits["SAFE_X"]   # [min, max] in metres
    SAFE_Y = limits["SAFE_Y"]
    SAFE_Z = limits["SAFE_Z"]
    margin = data.get("margin_m", 0.01)
    ts     = data.get("timestamp", "unknown")

    warn_m = WARNING_MM / 1000.0

    print(f"\n{BOLD}Safe Limits Loaded{RESET}  (recorded: {ts})")
    print(f"  X: [{SAFE_X[0]*1000:+.1f}, {SAFE_X[1]*1000:+.1f}] mm")
    print(f"  Y: [{SAFE_Y[0]*1000:+.1f}, {SAFE_Y[1]*1000:+.1f}] mm")
    print(f"  Z: [{SAFE_Z[0]*1000:+.1f}, {SAFE_Z[1]*1000:+.1f}] mm")
    print(f"  Margin already baked in: {margin*1000:.0f} mm")
    print(f"\nConnecting to robot at {ROBOT_IP}...")
    print(f"Polling at {POLL_HZ} Hz.  Ctrl-C to quit.\n")

    interval = 1.0 / POLL_HZ

    try:
        while True:
            t0   = time.time()
            pose = get_tcp_pose()

            lines = []
            lines.append(f"{BOLD}═══ UR10 Safe Workspace Monitor ═══{RESET}   {time.strftime('%H:%M:%S')}")
            lines.append("")

            if pose is None:
                lines.append(f"  {RED}Robot not connected — retrying...{RESET}")
            else:
                x, y, z = pose[0], pose[1], pose[2]

                lines.append(fmt_axis("X", x, SAFE_X[0], SAFE_X[1], warn_m))
                lines.append(fmt_axis("Y", y, SAFE_Y[0], SAFE_Y[1], warn_m))
                lines.append(fmt_axis("Z", z, SAFE_Z[0], SAFE_Z[1], warn_m))
                lines.append("")

                # Overall status
                violations = []
                if not (SAFE_X[0] <= x <= SAFE_X[1]):
                    violations.append(f"X={x*1000:+.1f}mm")
                if not (SAFE_Y[0] <= y <= SAFE_Y[1]):
                    violations.append(f"Y={y*1000:+.1f}mm")
                if not (SAFE_Z[0] <= z <= SAFE_Z[1]):
                    violations.append(f"Z={z*1000:+.1f}mm")

                if violations:
                    lines.append(f"  {RED}{BOLD}⚠  OUTSIDE SAFE ZONE: {', '.join(violations)}{RESET}")
                else:
                    warnings = []
                    for name, v, lo, hi in [("X", x, SAFE_X[0], SAFE_X[1]),
                                             ("Y", y, SAFE_Y[0], SAFE_Y[1]),
                                             ("Z", z, SAFE_Z[0], SAFE_Z[1])]:
                        dist = min(abs(v - lo), abs(v - hi)) * 1000
                        if dist < WARNING_MM:
                            warnings.append(f"{name} ({dist:.0f}mm to edge)")
                    if warnings:
                        lines.append(f"  {YELLOW}{BOLD}⚠  Near limit: {', '.join(warnings)}{RESET}")
                    else:
                        lines.append(f"  {GREEN}{BOLD}✓  All axes within safe zone{RESET}")

            # Redraw
            print(CLEAR + "\n".join(lines), end="", flush=True)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print(f"\n\n{RESET}Quit.")


if __name__ == "__main__":
    main()
