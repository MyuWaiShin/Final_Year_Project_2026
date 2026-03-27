"""
00_test_urp.py
==============
Test-bed for Dashboard gripper URP programs.
Toggles between open / close / timed-close to verify paths and behaviour.

Controls:
  O  → open_gripper.urp        (single open)
  C  → close_gripper.urp       (single close — initial grab)
  L  → close_gripper_timed.urp (loop close — slip monitoring)
  S  → stop current program
  W  → print sensor snapshot (raw voltage + mm, no calibration)
  R  → stream live readings every 0.5s (press Enter to stop)
  Q  → quit
"""

import socket
import struct
import threading
import time

# ── Config ────────────────────────────────────────────────────────────────
ROBOT_IP       = "192.168.8.102"
DASHBOARD_PORT = 29999
FEEDBACK_PORT  = 30002

PROGRAMS = {
    "O": "/programs/myu/open_gripper.urp",
    "C": "/programs/myu/close_gripper.urp",
    "L": "/programs/myu/close_gripper_timed.urp",
}

# RG2 uses TDI1 (Tool Digital Input 1) = bit 17 of the DI word for contact detection
DI_CONTACT_BIT = (1 << 17)


# ── Live sensor reader ─────────────────────────────────────────────────────
class SensorReader:
    def __init__(self, ip):
        self.latest_ai2 = 0.0
        self.latest_di  = 0        # 64-bit digital input word from type-3 packet
        self.running    = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((ip, FEEDBACK_PORT))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                header = self._sock.recv(4)
                if not header or len(header) < 4:
                    continue
                pkt_len  = struct.unpack("!I", header)[0]
                pkt_data = self._sock.recv(pkt_len - 4)
                offset = 1
                while offset < len(pkt_data):
                    if offset + 4 > len(pkt_data):
                        break
                    p_size = struct.unpack("!I", pkt_data[offset:offset+4])[0]
                    p_type = pkt_data[offset+4]

                    if p_type == 2 and offset + 15 <= len(pkt_data):
                        # Tool Data — analog input 0 = gripper width voltage
                        self.latest_ai2 = struct.unpack("!d", pkt_data[offset+7:offset+15])[0]

                    elif p_type == 3 and offset + 13 <= len(pkt_data):
                        # Digital I/O Data — 64-bit DI word at offset+5
                        # Bit 16 = TDI0 (gripper ready/powered)
                        # Bit 17 = TDI1 (RG2 contact / force limit reached)
                        self.latest_di = struct.unpack("!Q", pkt_data[offset+5:offset+13])[0]

                    if p_size == 0:
                        break
                    offset += p_size
            except Exception:
                pass

    def get_raw(self) -> tuple:
        """Returns (voltage_V, raw_mm, calibrated_mm).

        Tool flange analog input (TAI0) is always 0-5V on CB3.
        Max measured from RG2 = 3.7V (29kOhm UR divider / (10k+29k)).
        raw_mm     = (V / 3.7) * 110            <- no correction
        calibrated = raw_mm * slope + offset    <- two-point correction
        Calibration: raw~8.5mm->actual 10.5mm; raw~65.8mm->actual 91.0mm
        """
        voltage = max(self.latest_ai2, 0.0)
        raw_mm  = (voltage / 3.7) * 110.0
        slope   = (91.0 - 10.5) / (65.8 - 8.5)
        offset  = 10.5 - (8.5 * slope)
        cal_mm  = max(0.0, (raw_mm * slope) + offset)
        return round(voltage, 4), round(raw_mm, 1), round(cal_mm, 1)

    def contact(self) -> bool:
        return bool(self.latest_di & DI_CONTACT_BIT)

    def snapshot(self) -> str:
        voltage, raw_mm, cal_mm = self.get_raw()
        di_contact = self.contact()
        return (
            f"  V:{voltage:.4f}V"
            f"  | raw(3.7V): {raw_mm:5.1f}mm"
            f"  | calibrated: {cal_mm:5.1f}mm"
            f"  | DI8: {'HIGH' if di_contact else 'LOW'}"
        )

    def close(self):
        self.running = False
        try: self._sock.close()
        except Exception: pass


# ── Dashboard helper ───────────────────────────────────────────────────────
def dashboard(ip: str, cmd: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((ip, DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            resp = s.recv(1024).decode().strip()
            s.close()
            return resp
        except Exception as e:
            print(f"  [Dashboard] {cmd}: {e}  (attempt {attempt}/{retries})")
            if attempt < retries:
                time.sleep(0.5)
    return ""

def run_program(ip: str, key: str, path: str, sensor: SensorReader):
    print(f"\n>>> [{key}] {path}")
    print("  Stopping current program...")
    resp = dashboard(ip, "stop")
    print(f"  stop → {resp}")

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if "false" in dashboard(ip, "running", retries=1).lower():
            break
        time.sleep(0.2)

    print(f"  Loading...")
    resp = dashboard(ip, f"load {path}")
    print(f"  load → {resp}")
    time.sleep(0.3)

    print(f"  Playing...")
    resp = dashboard(ip, "play")
    print(f"  play → {resp}")

    # Wait and then print a sensor snapshot
    time.sleep(2.5)
    print(f"\n  --- Sensor after 2.5s ---")
    print(sensor.snapshot())
    print()

def stop_all(ip: str, sensor: SensorReader):
    print("\n>>> [S] Stopping...")
    resp = dashboard(ip, "stop")
    print(f"  stop → {resp}")
    time.sleep(0.5)
    print(sensor.snapshot())
    print()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  URP TOGGLE TESTER")
    print("=" * 55)
    print(f"  Robot IP: {ROBOT_IP}")
    print()
    print("  Programs:")
    for key, path in PROGRAMS.items():
        print(f"    [{key}]  {path}")
    print("    [W]  print sensor snapshot (once)")
    print("    [R]  stream live readings every 0.5s (Enter to stop)")
    print("    [S]  stop")
    print("    [Q]  quit")
    print("=" * 55 + "\n")

    print("Connecting to feedback port...")
    try:
        sensor = SensorReader(ROBOT_IP)
        time.sleep(1.0)   # wait for first packets
        print("Connected!\n")
        print("Initial sensor reading:")
        print(sensor.snapshot())
        print()
    except Exception as e:
        print(f"Could not connect: {e}")
        return

    while True:
        try:
            key = input("Command [O/C/L/W/S/Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break

        if key == "Q":
            stop_all(ROBOT_IP, sensor)
            break
        elif key == "S":
            stop_all(ROBOT_IP, sensor)
        elif key == "W":
            print(sensor.snapshot())
            print()
        elif key == "R":
            print("  Streaming... press Enter to stop.")
            import threading as _t
            _stop = _t.Event()
            def _stream():
                while not _stop.is_set():
                    print(sensor.snapshot())
                    time.sleep(0.5)
            th = _t.Thread(target=_stream, daemon=True)
            th.start()
            input()
            _stop.set()
            th.join(timeout=1.0)
            print("  Stopped streaming.\n")
        elif key in PROGRAMS:
            run_program(ROBOT_IP, key, PROGRAMS[key], sensor)
        else:
            print(f"  Unknown: '{key}'. Use O, C, L, W, R, S, Q.\n")

    sensor.close()
    print("Done.")


if __name__ == "__main__":
    main()
