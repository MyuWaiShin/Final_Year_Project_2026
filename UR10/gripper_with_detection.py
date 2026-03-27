"""
gripper_with_detection.py
==========================
UR10 + RG2 gripper controller.
Uses raw sockets only (no rtde, no URscript URCap restrictions).

Ports:
  29999  Dashboard Server  — load / play / stop  .urp programs
  30002  Secondary Client  — persistent state stream + URScript commands

URP paths on robot controller:
  URP_OPEN  = /programs/myu/open_gripper.urp
  URP_CLOSE = /programs/myu/close_gripper.urp

Consistent with 00_test_grip_urscript.py socket architecture.
"""

import socket
import struct
import time
import threading

# ── URP paths on the robot ────────────────────────────────────────────────────
URP_OPEN  = "/programs/myu/open_gripper.urp"
URP_CLOSE = "/programs/myu/close_gripper.urp"

DASHBOARD_PORT = 29999
SECONDARY_PORT = 30002


class UR10Controller:
    """
    Combined Dashboard + I/O Feedback controller for UR10 with RG2 Gripper.

    Architecture (consistent with 00_test_grip_urscript.py):
      - State stream: persistent socket on port 30002, buffered packet reading
      - URScript commands: persistent socket on port 30002 (separate connection)
      - Dashboard commands: one-shot connections on port 29999
    """

    def __init__(self, ip: str):
        self.ip = ip

        # Feedback state (updated by background thread)
        self.latest_digital_in = 0
        self.latest_analog_in2  = 0.0
        self.running = True

        # ── State stream (read-only) ──────────────────────────────────────
        print(f"[UR10] Connecting state stream ({ip}:{SECONDARY_PORT})...")
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._state_sock.settimeout(5.0)
        self._state_sock.connect((ip, SECONDARY_PORT))
        print("[UR10] State stream ready.")
        threading.Thread(target=self._state_loop, daemon=True, name="UR10State").start()

        # ── Command socket (persistent, for URScript movel etc.) ──────────
        print(f"[UR10] Connecting command socket ({ip}:{SECONDARY_PORT})...")
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._cmd_sock.settimeout(5.0)
        self._cmd_sock.connect((ip, SECONDARY_PORT))
        self._cmd_lock = threading.Lock()
        print("[UR10] Command socket ready.")
        threading.Thread(target=self._drain_cmd, daemon=True, name="UR10Drain").start()

    # ── State parsing ─────────────────────────────────────────────────────────

    def _state_loop(self):
        """Buffered packet reader — never loses partial reads."""
        buf = b""
        while self.running:
            try:
                chunk = self._state_sock.recv(4096)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buf += chunk
                while len(buf) >= 4:
                    plen = struct.unpack("!I", buf[:4])[0]
                    if len(buf) < plen:
                        break
                    self._parse_packet(buf[:plen])
                    buf = buf[plen:]
            except Exception:
                time.sleep(0.01)

    def _parse_packet(self, pkt: bytes):
        if len(pkt) < 5 or pkt[4] != 16:
            return
        offset = 5
        while offset + 5 <= len(pkt):
            sp_len  = struct.unpack("!I", pkt[offset:offset+4])[0]
            sp_type = pkt[offset+4]
            if sp_len == 0:
                break
            if sp_type == 2 and sp_len >= 15:   # Tool Data
                # AI2 voltage: 8-byte double at byte 7 of sub-packet
                self.latest_analog_in2 = struct.unpack("!d", pkt[offset+7:offset+15])[0]
                # DI byte: last byte of sub-packet
                self.latest_digital_in = pkt[offset + sp_len - 1]
            offset += sp_len

    def _drain_cmd(self):
        """Discard unsolicited data from command socket."""
        while self.running:
            try:
                self._cmd_sock.recv(4096)
            except Exception:
                time.sleep(0.01)

    # ── Dashboard commands (one-shot, consistent with 00_test_grip_urscript.py) ─

    def _dashboard(self, cmd: str) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((self.ip, DASHBOARD_PORT))
            s.recv(1024)   # consume greeting
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"[Dashboard] {cmd!r}: {e}")
            return ""

    def load_program(self, program_name: str) -> str:
        """Load a .urp by full path or name.
        If no '/' prefix, prepends /programs/myu/ automatically.
        """
        if not program_name.startswith("/"):
            program_name = f"/programs/myu/{program_name}"
        return self._dashboard(f"load {program_name}")

    def play(self) -> str:
        return self._dashboard("play")

    def stop(self) -> str:
        return self._dashboard("stop")

    # ── URScript motion (persistent socket) ───────────────────────────────────

    def send_urscript(self, script: str):
        """Send URScript via persistent command socket."""
        payload = (script.strip() + "\n").encode()
        with self._cmd_lock:
            try:
                self._cmd_sock.sendall(payload)
                return True
            except Exception as e:
                print(f"[URScript] send failed: {e}")
                return False

    def movel(self, pose, a=0.5, v=0.1):
        """Send movel command keeping existing TCP orientation."""
        x, y, z, rx, ry, rz = pose
        script = (
            f"movel(p[{x:.6f},{y:.6f},{z:.6f},"
            f"{rx:.6f},{ry:.6f},{rz:.6f}],a={a:.3f},v={v:.3f})"
        )
        return self.send_urscript(script)

    # ── Sensor API ────────────────────────────────────────────────────────────

    def get_width_mm(self) -> float:
        """Calibrated RG2 finger width from AI2 voltage.
        Calibration: raw 8.5mm→actual 10.5mm, raw 65.8mm→actual 91mm.
        """
        voltage  = max(self.latest_analog_in2, 0.0)
        raw_mm   = (voltage / 3.7) * 110.0
        slope    = (91.0 - 10.5) / (65.8 - 8.5)   # ≈ 1.405
        offset   = 10.5 - (8.5 * slope)             # ≈ -1.44
        return max(0.0, round(raw_mm * slope + offset, 1))

    def is_force_detected(self) -> bool:
        """True when DI8 HIGH — gripper jaws are touching something."""
        return (self.latest_digital_in & 0b00000001) != 0

    def is_object_detected(self) -> bool:
        """True when contact AND gripper spread > 12mm (object between jaws)."""
        if not self.is_force_detected():
            return False
        return self.get_width_mm() > 12.0

    def get_debug_info(self) -> dict:
        b = self.latest_digital_in
        return {
            "raw_byte" : b,
            "binary"   : format(b, "08b"),
            "bit0_DI8" : bool(b & 0b00000001),
            "bit1_DI9" : bool(b & 0b00000010),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self.running = False
        try: self._state_sock.close()
        except: pass
        try: self._cmd_sock.close()
        except: pass


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal, os
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    ROBOT_IP = "192.168.8.102"

    ctrl = UR10Controller(ROBOT_IP)
    print("Waiting for I/O feedback...")
    time.sleep(1.5)

    print("\n=== UR10 Gripper Control with Grasp Detection ===")
    print("  o = Open gripper")
    print("  c = Close gripper (with object detection)")
    print("  s = Status")
    print("  q = Quit\n")

    while True:
        cmd = input("Command (o/c/s/q): ").strip().lower()

        if cmd == "q":
            break

        elif cmd == "o":
            print("Opening gripper...")
            ctrl.load_program(URP_OPEN)
            time.sleep(0.3)
            ctrl.play()
            time.sleep(2.5)
            print(f"Done. Width: {ctrl.get_width_mm():.1f} mm\n")

        elif cmd == "c":
            print("Closing gripper...")
            ctrl.load_program(URP_CLOSE)
            time.sleep(0.3)
            ctrl.play()
            time.sleep(2.5)
            w   = ctrl.get_width_mm()
            v   = ctrl.latest_analog_in2
            frc = ctrl.is_force_detected()
            obj = ctrl.is_object_detected()
            print(f"\n--- Grasp Result ---")
            print(f"  Width   : {w:.1f} mm")
            print(f"  Voltage : {v:.3f} V")
            print(f"  Contact : {frc}")
            print(f"  Object  : {obj}")
            print(f"  Label   : {'HOLDING' if obj else 'EMPTY'}\n")

        elif cmd == "s":
            w = ctrl.get_width_mm()
            print(f"\n--- Status ---")
            print(f"  Width   : {w:.1f} mm")
            print(f"  Voltage : {ctrl.latest_analog_in2:.3f} V")
            print(f"  Contact : {ctrl.is_force_detected()}")
            print(f"  Object  : {ctrl.is_object_detected()}")
            print(f"  DI byte : {bin(ctrl.latest_digital_in)}\n")

        else:
            print("Unknown command.\n")

    ctrl.close()
    print("Done.")
