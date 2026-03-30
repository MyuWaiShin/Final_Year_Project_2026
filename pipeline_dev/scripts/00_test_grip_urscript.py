"""
15_test_grip_urscript.py
========================
Diagnostic: answers two questions:

  Q1. Does rg_grip() work from port 30002 raw URScript?
      (If YES: use it in the transit loop. If NO: dashboard URP is the only way.)

  Q2. Does a running dashboard URP survive when we send a movej via port 30002?
      (If YES: dashboard loop URP IS compatible with socket motion. 
       If NO: they conflict.)

Run from 13_03_2026\:
  python scripts/15_test_grip_urscript.py
"""

import socket
import time
import sys

ROBOT_IP       = "192.168.8.102"
URSCRIPT_PORT  = 30002
DASHBOARD_PORT = 29999

URP_OPEN        = "/programs/myu/open_gripper.urp"
URP_CLOSE       = "/programs/myu/close_gripper.urp"
URP_CLOSE_LOOP  = "/programs/myu/close_gripper_timed.urp"


# ── helpers ──────────────────────────────────────────────────────────────────

def dashboard(cmd: str) -> str:
    s = socket.socket()
    s.settimeout(5.0)
    s.connect((ROBOT_IP, DASHBOARD_PORT))
    s.recv(1024)
    s.sendall((cmd + "\n").encode())
    resp = s.recv(1024).decode().strip()
    s.close()
    return resp


def send_urscript(sock, script: str):
    sock.sendall((script.strip() + "\n").encode())


def connect_cmd():
    s = socket.socket()
    s.settimeout(10.0)
    s.connect((ROBOT_IP, URSCRIPT_PORT))
    return s


def drain_until(sock, seconds: float):
    """Read and discard data from sock for `seconds`."""
    deadline = time.time() + seconds
    sock.settimeout(0.1)
    while time.time() < deadline:
        try:
            sock.recv(4096)
        except socket.timeout:
            pass


# ── TEST 1: rg_grip via port 30002 one-liner ──────────────────────────────────

def test1_rg_grip_oneliner():
    print("\n" + "="*60)
    print("TEST 1: Single-line rg_grip(10, 0) via port 30002")
    print("  Expected: gripper moves slightly (towards 0mm, 10N force)")
    print("="*60)
    input("  Position gripper fully OPEN first, then press ENTER...")

    s = connect_cmd()
    print("  Sending: rg_grip(10, 0)")
    send_urscript(s, "rg_grip(10, 0)")
    print("  Waiting 3s — observe if gripper moves...")
    drain_until(s, 3.0)
    s.close()
    result = input("  Did gripper move? (y/n): ").strip().lower()
    return result == "y"


# ── TEST 2: rg_grip inside a def program ─────────────────────────────────────

def test2_rg_grip_def():
    print("\n" + "="*60)
    print("TEST 2: rg_grip inside URScript def block via port 30002")
    print("="*60)
    input("  Ensure gripper is OPEN, then press ENTER...")

    s = connect_cmd()
    prog = (
        "def test_grip():\n"
        "  rg_grip(10, 0)\n"
        "end\n"
        "test_grip()\n"
    )
    print("  Sending multi-line URScript program...")
    send_urscript(s, prog)
    drain_until(s, 3.0)
    s.close()
    result = input("  Did gripper move? (y/n): ").strip().lower()
    return result == "y"


# ── TEST 3: dashboard loop URP + socket movej coexistence ─────────────────────

def test3_urp_plus_movej():
    print("\n" + "="*60)
    print("TEST 3: Dashboard loop URP + socket movej coexistence")
    print("  - Opens gripper via dashboard")
    print("  - Starts close_gripper_timed.urp (loops every ~0.2s)")
    print("  - Immediately sends a movej to a NEARBY safe position via socket")
    print("  - Observes if gripper closes DURING the movej")
    print("="*60)
    print("  WARNING: robot will move! Make sure workspace is clear.")
    input("  Press ENTER when ready...")

    # Get current pose first via state socket
    state_s = socket.socket()
    state_s.settimeout(5.0)
    state_s.connect((ROBOT_IP, URSCRIPT_PORT))
    time.sleep(0.3)

    # 1. Open gripper
    print("  [1] Opening gripper...")
    dashboard(f"load {URP_OPEN}")
    time.sleep(0.2)
    dashboard("play")
    time.sleep(3.0)

    # 2. Start loop close URP
    print("  [2] Starting close_gripper_timed.urp...")
    dashboard("stop")
    time.sleep(0.1)
    dashboard(f"load {URP_CLOSE_LOOP}")
    time.sleep(0.2)
    dashboard("play")
    time.sleep(0.1)   # tiny settle, then immediately send movej

    # 3. Send a tiny movej (move 5mm in Z) while loop URP running
    cmd_s = connect_cmd()
    print("  [3] Sending movej (+5mm Z) while loop URP running...")
    # NOTE: sends movej — if loop URP dies, gripper won't move
    send_urscript(cmd_s, "movej(p[0.3889, -0.8252, -0.2762, 1.5913, 2.6109, -0.0442], a=0.2, v=0.2)")
    drain_until(cmd_s, 5.0)
    cmd_s.close()
    state_s.close()

    # 4. Stop loop URP if still running
    print("  [4] Stopping loop URP...")
    dashboard("stop")
    time.sleep(0.5)

    result = input("  Did gripper CLOSE during the movej? (y/n): ").strip().lower()
    return result == "y"


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRG2 Grip Loop Diagnostic Tool")
    print("Robot:", ROBOT_IP)
    print("\nThis will test three things and report results.")
    print("Ctrl-C to abort at any time.\n")

    results = {}

    try:
        results["Q1: rg_grip one-liner (port 30002)"]  = test1_rg_grip_oneliner()
        results["Q2: rg_grip in def block (port 30002)"] = test2_rg_grip_def()
        results["Q3: loop URP survives socket movej"]   = test3_urp_plus_movej()
    except KeyboardInterrupt:
        print("\nAborted.")

    print("\n" + "="*60)
    print("RESULTS:")
    for q, r in results.items():
        print(f"  {'YES' if r else 'NO ':3s} — {q}")
    print("="*60)
    print("\nInterpretation:")
    q1 = results.get("Q1: rg_grip one-liner (port 30002)", False)
    q2 = results.get("Q2: rg_grip in def block (port 30002)", False)
    q3 = results.get("Q3: loop URP survives socket movej", False)

    if q1 and q2:
        print("  rg_grip IS available in port 30002 URScript.")
        print("  -> movej_with_grip_loop() should work. Check URScript syntax.")
    elif not q1 and not q2:
        print("  rg_grip is NOT available in port 30002 URScript (URCap restriction).")
        if q3:
            print("  -> GOOD: dashboard loop URP SURVIVES socket movej!")
            print("  -> Fix: keep loop URP running, don't call dashboard('stop') until ready to open.")
        else:
            print("  -> BAD: dashboard loop URP is KILLED by socket movej.")
            print("  -> Options:")
            print("     a) Use intermediate stops: movej half-way, re-close, continue.")
            print("     b) Rely on RG2's own force control (40N holds without re-closing).")
    elif q3:
        print("  Dashboard loop URP survives socket movej -> use dashboard URP approach.")
    else:
        print("  Neither approach works reliably on this setup.")
        print("  -> Rely on RG2 force control + Python slip detection only.")
