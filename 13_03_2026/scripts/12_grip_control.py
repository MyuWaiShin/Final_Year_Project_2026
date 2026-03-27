"""
grip_control_myu.py
===================
Adapted version of grip_control_with_slip_detection.py for Myu's UR10 config.

Robot IP  : 192.168.8.102
URP paths :
  Close (loop) : /programs/myu/close_gripper_timed.urp
                 *** THIS URP MUST BE A LOOP PROGRAM ON THE TEACH PENDANT ***
                 Structure: Loop { RG2(0, 40N)  Wait 0.1s }
                 If it's currently a one-shot, edit it on the pendant to add
                 a Loop wrapper around the RG2 command.
  Open         : /programs/myu/open_gripper.urp

DI8 parsing: reads last byte of Tool Data sub-packet (offset + p_size - 1),
             checks bit 0 (LSB) for force-limit contact signal.

AI2 parsing: 8-byte double at offset+7 in Tool Data sub-packet → gripper width.
"""

import socket
import struct
import time
import threading


# ── Config ────────────────────────────────────────────────────────────────────

ROBOT_IP  = "192.168.8.102"
URP_CLOSE = "/programs/myu/close_gripper_timed.urp"  # MUST be a Loop program!
URP_OPEN  = "/programs/myu/open_gripper.urp"

DASHBOARD_PORT = 29999
FEEDBACK_PORT  = 30002

WIDTH_CLOSED_MM = 11.0   # below this = fully closed / no object
WIDTH_OBJECT_MM = 12.0   # above this = holding object


# ── Controller ────────────────────────────────────────────────────────────────

class GripController:
    """
    Gripper-only controller for Myu's UR10 + RG2 setup.

    Requires grip_close.urp to be a Loop program on the robot:
       Loop:
         RG2 grip (width=0, force=40N)
         Wait 0.1s

    This loops forever, continuously re-gripping at 40N.
    Stop is sent before loading open_gripper.urp.
    """

    SLIP_CHECK_INTERVAL = 0.3

    def __init__(self, ip=ROBOT_IP):
        self.ip = ip

        # Sensor values (updated by feedback thread)
        self.latest_ai2 = 0.0
        self.latest_di  = 0    # last byte of Tool Data (DI byte)
        self.running    = True

        # State machine
        self._grip_state      = "open"
        self._grip_lock       = threading.Lock()
        self._had_object      = False
        self._monitoring_active = False
        self._loop_stopped    = False

        # Start feedback stream
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((ip, FEEDBACK_PORT))
        print(f"[Init] Connected to feedback stream ({ip}:{FEEDBACK_PORT})")
        threading.Thread(target=self._feedback_loop, daemon=True, name="Feedback").start()

        # Start slip monitor
        threading.Thread(target=self._slip_monitor, daemon=True, name="SlipMon").start()

    # ── Feedback parsing ─────────────────────────────────────────────────────

    def _feedback_loop(self):
        """Parse secondary state stream for AI2 (width) and DI (force contact)."""
        while self.running:
            try:
                header = self._sock.recv(4)
                if not header or len(header) < 4:
                    continue
                pkt_len  = struct.unpack("!I", header)[0]
                pkt_data = self._sock.recv(pkt_len - 4)

                offset = 1  # skip outer message-type byte
                while offset < len(pkt_data):
                    if offset + 4 > len(pkt_data):
                        break
                    p_size = struct.unpack("!I", pkt_data[offset:offset+4])[0]
                    if p_size == 0:
                        break
                    p_type = pkt_data[offset + 4]

                    if p_type == 2:  # Tool Data sub-packet
                        # AI2 — gripper width voltage — 8-byte double at byte 7
                        if offset + 15 <= len(pkt_data):
                            self.latest_ai2 = struct.unpack(
                                "!d", pkt_data[offset+7:offset+15])[0]
                        # DI byte — last byte of sub-packet
                        if offset + p_size <= len(pkt_data):
                            self.latest_di = pkt_data[offset + p_size - 1]

                    offset += p_size
            except Exception:
                pass

    # ── Sensor API ────────────────────────────────────────────────────────────

    def get_width_mm(self) -> float:
        """Calibrated gripper width from AI2 voltage."""
        v      = max(self.latest_ai2, 0.0)
        raw_mm = (v / 3.7) * 110.0
        slope  = (91.0 - 10.5) / (65.8 - 8.5)   # ≈ 1.405
        offset = 10.5 - (8.5 * slope)             # ≈ -1.44
        return max(0.0, round(raw_mm * slope + offset, 1))

    def is_contact(self) -> bool:
        """True when DI8 HIGH — gripper jaws are touching something."""
        return (self.latest_di & 0b00000001) != 0

    def is_holding(self) -> bool:
        """True when contact reached AND fingers are spread (object present)."""
        return self.is_contact() and self.get_width_mm() > WIDTH_OBJECT_MM

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def _cmd(self, cmd: str) -> str:
        try:
            s = socket.socket()
            s.settimeout(3.0)
            s.connect((self.ip, DASHBOARD_PORT))
            s.recv(1024)
            s.sendall((cmd + "\n").encode())
            r = s.recv(1024).decode().strip()
            s.close()
            return r
        except Exception as e:
            print(f"  [Dashboard] {cmd!r}: {e}")
            return ""

    def _stop_and_wait(self, timeout=4.0):
        """Send stop, poll until not running, then drain RG2 buffer."""
        self._cmd("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "false" in self._cmd("running").lower():
                break
            time.sleep(0.15)
        time.sleep(1.5)   # drain RG2 internal command buffer

    def _play_urp(self, path: str):
        self._stop_and_wait()
        r = self._cmd(f"load {path}")
        print(f"  [Dashboard] load: {r!r}")
        time.sleep(0.2)
        r = self._cmd("play")
        print(f"  [Dashboard] play: {r!r}")

    # ── Slip monitor ──────────────────────────────────────────────────────────

    def _slip_monitor(self):
        while self.running:
            with self._grip_lock:
                state = self._grip_state

            if state == "closing" and self._monitoring_active:
                contact = self.is_contact()
                width   = self.get_width_mm()

                # Confirm holding object
                if contact and width > WIDTH_OBJECT_MM and not self._had_object:
                    self._had_object = True
                    print(f"  [Grip] Holding at {width:.1f} mm (loop running)")

                # Detect slip: was holding, now fully closed / empty
                if contact and width < WIDTH_CLOSED_MM and not self._loop_stopped:
                    if self._had_object:
                        print(f"\n  [SLIP] Object dropped! Width {width:.1f} mm — re-closing...")
                        # Loop URP will auto-re-close; mark and re-enable monitoring
                    else:
                        print(f"  [Grip] Fully closed at {width:.1f} mm — stopping loop")
                    self._cmd("stop")
                    self._loop_stopped = True
                    self._had_object   = False

            elif state != "closing":
                self._had_object        = False
                self._monitoring_active = False
                self._loop_stopped      = False

            time.sleep(self.SLIP_CHECK_INTERVAL)

    # ── Public API ────────────────────────────────────────────────────────────

    def close_grip(self):
        """
        Start the loop close URP. Returns immediately.
        Slip detection activates after 2.5s settle.
        """
        with self._grip_lock:
            self._grip_state = "closing"
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False

        print("Closing gripper (loop URP)...")
        self._play_urp(URP_CLOSE)

        def _enable_mon():
            time.sleep(2.5)
            with self._grip_lock:
                if self._grip_state == "closing":
                    self._monitoring_active = True
        threading.Thread(target=_enable_mon, daemon=True).start()

    def open_grip(self):
        """Stop loop URP and open gripper."""
        with self._grip_lock:
            self._grip_state = "open"
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False
        print("Opening gripper...")
        self._play_urp(URP_OPEN)

    def status(self):
        width = self.get_width_mm()
        print(f"\n--- Grip Status ---")
        print(f"  State    : {self._grip_state}")
        print(f"  Width    : {width:.1f} mm")
        print(f"  Voltage  : {self.latest_ai2:.3f} V")
        print(f"  Contact  : {self.is_contact()}")
        print(f"  Holding  : {self.is_holding()}")
        print(f"  Monitor  : {self._monitoring_active}")
        print(f"  DI byte  : {bin(self.latest_di)}")
        print(f"-------------------\n")

    def close(self):
        self.running = False
        try: self._sock.close()
        except: pass


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal, os
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    print(f"Connecting to {ROBOT_IP}...")
    ctrl = GripController(ROBOT_IP)
    print("Stabilising sensors...")
    time.sleep(1.5)

    print("\n=== Grip Control + Slip Detection ===")
    print("  c = Close (loop URP — grips continuously)")
    print("  o = Open")
    print("  s = Status")
    print("  q = Quit")
    print(f"\nNOTE: {URP_CLOSE} MUST be a Loop program on the robot.")
    print("      Structure: Loop { RG2(0, 40N)  Wait 0.1s }\n")

    while True:
        cmd = input("Command (c/o/s/q): ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "c":
            ctrl.close_grip()
            time.sleep(2.5)
            ctrl.status()
        elif cmd == "o":
            ctrl.open_grip()
            time.sleep(2.5)
            ctrl.status()
        elif cmd == "s":
            ctrl.status()
        else:
            print("Unknown command.\n")

    ctrl.close()
