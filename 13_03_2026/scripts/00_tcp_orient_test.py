"""
tcp_orient_test.py
==================
Sends a URScript command to reorient the TCP in-place (no XYZ movement)
so the gripper approaches from directly ABOVE (Z-axis pointing straight down).

The orientation is expressed as a rotation vector [rx, ry, rz].
For "pointing straight down" on a standard UR arm:
  rx = pi  (180 deg around X) keeps the tool Z-axis pointing -world_Z
  ry = 0
  rz = 0

Run this to test orientation only — robot arm must be clear of obstacles.
Edit APPROACH_Z if you also want to move the gripper height.

Usage:
    python tcp_orient_test.py

Commands in the prompt:
    t  = reorient TCP to point straight down (keep current XYZ)
    s  = show current TCP pose
    q  = quit
"""

import socket
import struct
import threading
import time
import math

ROBOT_IP      = "192.168.8.102"
URSCRIPT_PORT = 30002

# Rotation vector for "gripper pointing straight down"
# Adjust these if the tool is mounted at a different angle.
# [pi, 0, 0] = rotate 180 deg around base X → tool Z points down
TARGET_RX = math.pi    # 3.14159...
TARGET_RY = 0.0
TARGET_RZ = 0.0

MOVE_SPEED = 0.05   # m/s  — slow, safe
MOVE_ACCEL = 0.05   # m/s²


# ── Minimal robot socket ──────────────────────────────────────────────────────

class Robot:
    def __init__(self, ip):
        self.ip = ip
        self._tcp = [0.0] * 6
        self._lock = threading.Lock()
        self.running = True

        self._state = socket.socket()
        self._state.settimeout(5.0)
        self._state.connect((ip, URSCRIPT_PORT))
        threading.Thread(target=self._state_loop, daemon=True).start()

        self._cmd = socket.socket()
        self._cmd.settimeout(5.0)
        self._cmd.connect((ip, URSCRIPT_PORT))
        threading.Thread(target=self._drain, daemon=True).start()

        time.sleep(0.5)   # wait for first state packet

    def _state_loop(self):
        buf = b""
        while self.running:
            try:
                buf += self._state.recv(4096)
                while len(buf) >= 4:
                    plen = struct.unpack("!I", buf[:4])[0]
                    if len(buf) < plen:
                        break
                    self._parse(buf[:plen])
                    buf = buf[plen:]
            except Exception:
                time.sleep(0.01)

    def _parse(self, pkt):
        if len(pkt) < 5 or pkt[4] != 16:
            return
        off = 5
        while off + 5 <= len(pkt):
            slen = struct.unpack("!I", pkt[off:off+4])[0]
            if slen == 0:
                break
            if pkt[off+4] == 4 and slen >= 5 + 48:   # CartesianInfo
                vals = struct.unpack("!6d", pkt[off+5:off+5+48])
                with self._lock:
                    self._tcp = list(vals)
            off += slen

    def _drain(self):
        while self.running:
            try:
                self._cmd.recv(4096)
            except Exception:
                time.sleep(0.01)

    def pose(self):
        with self._lock:
            return list(self._tcp)

    def send(self, script):
        self._cmd.sendall((script.strip() + "\n").encode())

    def movel_to(self, x, y, z, rx, ry, rz, v=MOVE_SPEED, a=MOVE_ACCEL, tol=0.003, timeout=20.0):
        cmd = f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],a={a},v={v})"
        self.send(cmd)
        deadline = time.time() + timeout
        while time.time() < deadline:
            p = self.pose()
            dist = ((p[0]-x)**2 + (p[1]-y)**2 + (p[2]-z)**2)**0.5
            if dist < tol:
                return True
            time.sleep(0.02)
        return False

    def close(self):
        self.running = False
        try: self._state.close()
        except: pass
        try: self._cmd.close()
        except: pass


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal, os
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    print(f"Connecting to {ROBOT_IP}...")
    robot = Robot(ROBOT_IP)

    p = robot.pose()
    print(f"\nCurrent TCP:")
    print(f"  XYZ : {p[0]*1000:.1f}  {p[1]*1000:.1f}  {p[2]*1000:.1f} mm")
    print(f"  RxRyRz: {math.degrees(p[3]):.1f}  {math.degrees(p[4]):.1f}  {math.degrees(p[5]):.1f} deg")

    print(f"\nTarget orientation (straight down):")
    print(f"  RxRyRz: {math.degrees(TARGET_RX):.1f}  {math.degrees(TARGET_RY):.1f}  {math.degrees(TARGET_RZ):.1f} deg")

    print("\nCommands:  t = reorient to straight down | s = show pose | q = quit\n")

    while True:
        cmd = input("Command: ").strip().lower()

        if cmd == "q":
            break

        elif cmd == "s":
            p = robot.pose()
            print(f"\n  XYZ    : {p[0]*1000:.1f}  {p[1]*1000:.1f}  {p[2]*1000:.1f} mm")
            print(f"  RxRyRz : {math.degrees(p[3]):.2f}  {math.degrees(p[4]):.2f}  {math.degrees(p[5]):.2f} deg\n")

        elif cmd == "t":
            p = robot.pose()
            print(f"\n  Current pose: XYZ=({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}) mm")
            print(f"  Keeping XYZ, changing orientation to RxRyRz=({math.degrees(TARGET_RX):.1f}, 0, 0) deg")
            print(f"  Moving at {MOVE_SPEED*100:.0f} cm/s ... (hand on E-stop)\n")

            ok = robot.movel_to(p[0], p[1], p[2], TARGET_RX, TARGET_RY, TARGET_RZ)
            if ok:
                p2 = robot.pose()
                print(f"  Done! New RxRyRz: {math.degrees(p2[3]):.2f}  {math.degrees(p2[4]):.2f}  {math.degrees(p2[5]):.2f} deg\n")
            else:
                print("  Timeout — robot may not have reached target. Check pose with 's'.\n")

        else:
            print("Unknown command.\n")

    robot.close()
    print("Done.")
