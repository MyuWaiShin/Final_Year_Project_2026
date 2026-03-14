"""
show_safe_limits.py
====================
Live terminal viewer — loads Perception/safe_limits.json and streams the
current robot TCP position against the recorded workspace limits.

  Green  = inside safe zone
  Yellow = within WARNING_MM of a limit
  Red    = outside the limit (VIOLATION)

Usage:
    python UR10/show_safe_limits.py

Run in a second terminal while jogging the robot with the pendant.
Press Ctrl-C to quit.
"""

import json
import socket
import struct
import time
from pathlib import Path

ROBOT_IP    = "192.168.8.102"
LIMITS_PATH = Path(__file__).parent.parent / "Perception" / "safe_limits.json"
WARNING_MM  = 20.0
POLL_HZ     = 10

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
RESET  = "\033[0m";  BOLD   = "\033[1m";  CLEAR = "\033[2J\033[H"
BAR_W  = 28


def get_tcp_pose(ip=ROBOT_IP):
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


def _bar(value, lo, hi):
    """ASCII bar showing position within [lo, hi]."""
    if hi <= lo:
        return "[" + "?" * BAR_W + "]"
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    pos  = int(frac * BAR_W)
    return "[" + "─" * pos + "●" + "─" * (BAR_W - pos) + "]"


def fmt_axis(name, v, lo, hi, warn_m):
    """Returns a single coloured line for one axis."""
    v_mm  = v  * 1000
    lo_mm = lo * 1000
    hi_mm = hi * 1000

    in_range = (lo <= v <= hi)
    dist_mm  = min(abs(v - lo), abs(v - hi)) * 1000

    if not in_range:
        col, tag = RED,    "VIOLATION"
    elif dist_mm < warn_m * 1000:
        col, tag = YELLOW, "WARNING  "
    else:
        col, tag = GREEN,  "OK       "

    bar = _bar(v, lo, hi)
    return (
        f"  {col}{BOLD}{name}{RESET}  "
        f"current = {v_mm:+8.1f} mm   "
        f"range [{lo_mm:+7.1f} .. {hi_mm:+7.1f}]   "
        f"{col}{tag}{RESET}   "
        f"edge: {dist_mm:5.1f} mm   {bar}"
    )


def main():
    if not LIMITS_PATH.exists():
        print(f"ERROR: {LIMITS_PATH} not found.")
        print("Run:  python Perception/record_safe_limits.py  first.")
        return

    with open(LIMITS_PATH) as f:
        data = json.load(f)

    lim    = data["safe_limits"]
    # Sort so we always have (min, max) regardless of recording direction
    safe_x = tuple(sorted(lim["SAFE_X"]))
    safe_y = tuple(sorted(lim["SAFE_Y"]))
    safe_z = tuple(sorted(lim["SAFE_Z"]))
    margin = data.get("margin_m", 0.01)
    ts     = data.get("timestamp", "unknown")

    warn_m = WARNING_MM / 1000.0
    iv     = 1.0 / POLL_HZ

    print(f"\n{BOLD}Limits loaded{RESET}  (recorded: {ts}, margin={margin*1000:.0f} mm)")
    print(f"  X [{safe_x[0]*1000:+.1f} .. {safe_x[1]*1000:+.1f}]   "
          f"Y [{safe_y[0]*1000:+.1f} .. {safe_y[1]*1000:+.1f}]   "
          f"Z [{safe_z[0]*1000:+.1f} .. {safe_z[1]*1000:+.1f}] mm")
    print(f"Ctrl-C to quit.\n")

    try:
        while True:
            t0   = time.time()
            pose = get_tcp_pose()

            lines = [
                f"{BOLD}═══ UR10 Safe Workspace Monitor ═══{RESET}   {time.strftime('%H:%M:%S')}",
                "",
            ]

            if pose is None:
                lines.append(f"  {RED}Robot not connected — retrying...{RESET}")
            else:
                x, y, z = pose[0], pose[1], pose[2]

                lines.append(fmt_axis("X", x, safe_x[0], safe_x[1], warn_m))
                lines.append(fmt_axis("Y", y, safe_y[0], safe_y[1], warn_m))
                lines.append(fmt_axis("Z", z, safe_z[0], safe_z[1], warn_m))
                lines.append("")

                viols = []
                if not (safe_x[0] <= x <= safe_x[1]): viols.append(f"X={x*1000:+.1f}mm")
                if not (safe_y[0] <= y <= safe_y[1]): viols.append(f"Y={y*1000:+.1f}mm")
                if not (safe_z[0] <= z <= safe_z[1]): viols.append(f"Z={z*1000:+.1f}mm")

                if viols:
                    lines.append(f"  {RED}{BOLD}⚠  OUTSIDE SAFE ZONE: {', '.join(viols)}{RESET}")
                else:
                    warns = [ax for ax, v, lo, hi in
                             [("X",x,safe_x[0],safe_x[1]),
                              ("Y",y,safe_y[0],safe_y[1]),
                              ("Z",z,safe_z[0],safe_z[1])]
                             if min(abs(v-lo), abs(v-hi))*1000 < WARNING_MM]
                    if warns:
                        lines.append(f"  {YELLOW}{BOLD}⚠  Near limit: {', '.join(warns)}{RESET}")
                    else:
                        lines.append(f"  {GREEN}{BOLD}✓  All axes within safe zone{RESET}")

            print(CLEAR + "\n".join(lines), end="", flush=True)
            time.sleep(max(0.0, iv - (time.time() - t0)))

    except KeyboardInterrupt:
        print(f"\n\n{RESET}Quit.")


if __name__ == "__main__":
    main()
