"""
06_tcp_test.py
--------------
Live TCP reader with key-press save.

Controls:
  S        → save current TCP position (prompts for a label)
  Q / done → finish and write all saved positions to JSON file
"""

import json
import msvcrt
import time
from pathlib import Path

import rtde_receive

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
BASE_DIR    = SCRIPT_DIR.parent
OUTPUT_FILE = BASE_DIR / "data/saved_positions.json"

ROBOT_IP = "192.168.8.102"

# ── Connect ───────────────────────────────────────────────────────
print(f"Connecting to {ROBOT_IP}...")
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("Connected!\n")
print("=" * 55)
print("  S   = save current TCP position")
print("  Q   = quit and save file")
print("=" * 55)
print()

saved = {}

def read_tcp():
    tcp = rtde_r.getActualTCPPose()
    return tcp

def print_tcp(tcp, label=None):
    prefix = f"[{label}] " if label else "Current TCP: "
    print(f"{prefix}X={tcp[0]:.4f}  Y={tcp[1]:.4f}  Z={tcp[2]:.4f}"
          f"  RX={tcp[3]:.4f}  RY={tcp[4]:.4f}  RZ={tcp[5]:.4f}")

# ── Show live TCP until key pressed ───────────────────────────────
last_print = 0.0
while True:
    now = time.time()
    if now - last_print > 0.3:   # refresh every 300ms
        tcp = read_tcp()
        print(f"\rX={tcp[0]:.4f}  Y={tcp[1]:.4f}  Z={tcp[2]:.4f}   "
              f"[S=save  Q=quit]    ", end="", flush=True)
        last_print = now

    if msvcrt.kbhit():
        key = msvcrt.getwch().lower()

        if key == 's':
            tcp = read_tcp()
            print()   # newline after the live display
            label = input("  Enter label for this position: ").strip()
            if label:
                saved[label] = {
                    "x":  tcp[0], "y":  tcp[1], "z":  tcp[2],
                    "rx": tcp[3], "ry": tcp[4], "rz": tcp[5]
                }
                print_tcp(tcp, label=label)
                print(f"  ✓ Saved as '{label}'  ({len(saved)} total)\n")
            else:
                print("  (no label entered, skipped)\n")

        elif key in ('q', '\r'):
            print()
            break

        time.sleep(0.05)   # debounce

# ── Write JSON ────────────────────────────────────────────────────
if saved:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file to merge, not overwrite
    existing = {}
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing.update(saved)   # new entries overwrite same-named ones
    OUTPUT_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print(f"\nSaved {len(saved)} position(s) to {OUTPUT_FILE}")
    print("\nAll saved positions:")
    for lbl, pos in existing.items():
        print(f"  {lbl}: X={pos['x']:.4f}  Y={pos['y']:.4f}  Z={pos['z']:.4f}")
else:
    print("\nNo positions saved.")
