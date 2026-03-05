"""
record_safe_limits.py
=====================
Jog the robot to each axis extreme of your safe workspace,
press the corresponding key to record it, then press 'S' to
save and print the SAFE_X / SAFE_Y / SAFE_Z constants ready
to paste into pick_detected_object.py.

STEPS:
  1 — Jog to the LEFTMOST X position (left limit)
  2 — Jog to the RIGHTMOST X position (right limit)
  3 — Jog to the FURTHEST FORWARD Y position (max reach toward table)
  4 — Jog to the FURTHEST BACK Y position (nearest to robot base)
  5 — Jog to the LOWEST Z position (just above mat surface — where gripper descends)
  6 — Jog to the HIGHEST Z position (max height)

  S — Save limits and print the constants
  Q — Quit without saving

NOTE: All values are in METRES, in the robot BASE frame.
"""

import socket
import struct
import time
import json
from pathlib import Path

ROBOT_IP  = "192.168.8.102"
SAVE_PATH = Path("safe_limits.json")


def get_tcp_pose(ip=ROBOT_IP):
    """Returns [X, Y, Z, Rx, Ry, Rz] in metres/radians, or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((ip, 30003))
            data = s.recv(1100)
            if len(data) < 444 + 48:
                return None
            return list(struct.unpack("!6d", data[444:492]))
    except Exception as e:
        print(f"[Robot] {e}")
        return None


LABELS = {
    '1': "LEFT   (min X)",
    '2': "RIGHT  (max X)",
    '3': "FORWARD (min Y — furthest reach toward table)",
    '4': "BACK   (max Y — closest to robot base)",
    '5': "LOW    (min Z — just above mat)",
    '6': "HIGH   (max Z — max lift height)",
}

KEYS = ['1', '2', '3', '4', '5', '6']


def print_status(recorded):
    print("\n" + "=" * 58)
    print(f"  {'Step':<6} {'Description':<40} {'Recorded':>8}")
    print("  " + "-" * 54)
    for k in KEYS:
        rec = recorded.get(k)
        desc = LABELS[k]
        if rec:
            val = f"({rec[0]*1000:.0f}, {rec[1]*1000:.0f}, {rec[2]*1000:.0f}) mm"
            print(f"  [{k}]    {desc:<40}  ✓ {val}")
        else:
            print(f"  [{k}]    {desc:<40}  — not recorded")
    print("=" * 58)
    print("  S = Save & print constants    Q = Quit")
    print("=" * 58)


def print_live(pose, recorded):
    if pose:
        x, y, z = pose[0]*1000, pose[1]*1000, pose[2]*1000
        done = sum(1 for k in KEYS if k in recorded)
        print(f"\r  TCP:  X={x:8.1f}  Y={y:8.1f}  Z={z:8.1f} mm   [{done}/6 recorded]  ", end="", flush=True)
    else:
        print(f"\r  TCP: not connected                                    ", end="", flush=True)


def main():
    print("\n" + "=" * 58)
    print("  UR10 Safe Workspace Limit Recorder")
    print("=" * 58)
    print("  Jog the robot to each extreme, then press the key.")
    print("  Current TCP position is shown live below.\n")

    recorded = {}

    import sys
    import select

    # Use a simple input-per-step approach (no cv2 needed)
    try:
        import msvcrt  # Windows
        def get_key():
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8', errors='ignore').lower()
            return None
    except ImportError:
        import tty, termios
        def get_key():
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    return sys.stdin.read(1).lower()
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print_status(recorded)

    while True:
        time.sleep(0.1)
        pose = get_tcp_pose()
        print_live(pose, recorded)

        key = get_key()
        if key is None:
            continue

        if key in KEYS:
            tcp = get_tcp_pose()
            if tcp is None:
                print(f"\n  [!] Cannot read robot pose — check connection")
                continue
            recorded[key] = tcp
            x, y, z = tcp[0]*1000, tcp[1]*1000, tcp[2]*1000
            print(f"\n  ✓ [{key}] {LABELS[key]} recorded:"
                  f"  X={x:.1f}  Y={y:.1f}  Z={z:.1f} mm")
            print_status(recorded)

        elif key == 's':
            if len(recorded) < 6:
                missing = [k for k in KEYS if k not in recorded]
                print(f"\n  [!] Not all limits recorded yet. Missing: {', '.join(missing)}")
                continue

            # Extract limits
            safe_x_min = recorded['1'][0]
            safe_x_max = recorded['2'][0]
            safe_y_min = recorded['3'][1]
            safe_y_max = recorded['4'][1]
            safe_z_min = recorded['5'][2]
            safe_z_max = recorded['6'][2]

            # Add a small margin (10mm) so the robot doesn't scrape the exact edge
            MARGIN = 0.01

            print("\n" + "=" * 58)
            print("  SAFE WORKSPACE LIMITS")
            print("=" * 58)
            print(f"  X: {safe_x_min*1000:.1f} mm  →  {safe_x_max*1000:.1f} mm")
            print(f"  Y: {safe_y_min*1000:.1f} mm  →  {safe_y_max*1000:.1f} mm")
            print(f"  Z: {safe_z_min*1000:.1f} mm  →  {safe_z_max*1000:.1f} mm")
            print()
            print("  ── Paste this into pick_detected_object.py ──────────")
            print(f"  SAFE_X = ({safe_x_min + MARGIN:.4f},  {safe_x_max - MARGIN:.4f})  # left / right (m)")
            print(f"  SAFE_Y = ({safe_y_min + MARGIN:.4f},  {safe_y_max - MARGIN:.4f})  # forward / back (m)")
            print(f"  SAFE_Z = ({safe_z_min + MARGIN:.4f},  {safe_z_max - MARGIN:.4f})  # low / high (m)")
            print("=" * 58)

            # Save raw recorded poses to JSON
            data = {
                "recorded_points": {k: recorded[k] for k in KEYS},
                "safe_limits": {
                    "SAFE_X": [safe_x_min + MARGIN, safe_x_max - MARGIN],
                    "SAFE_Y": [safe_y_min + MARGIN, safe_y_max - MARGIN],
                    "SAFE_Z": [safe_z_min + MARGIN, safe_z_max - MARGIN],
                },
                "margin_m": MARGIN,
                "timestamp": time.ctime()
            }
            with open(SAVE_PATH, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"\n  Saved to {SAVE_PATH}")
            break

        elif key == 'q':
            print("\n  Quit — nothing saved.")
            break


if __name__ == "__main__":
    main()
