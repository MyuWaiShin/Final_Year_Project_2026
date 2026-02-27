import socket
import struct
import time
import threading


class UR10Controller:
    """
    UR10 + RG2 Gripper controller.

    GRIPPER CONTROL:
    Uses Dashboard Server (port 29999) to load and play .urp programs.
    rg_grip() is a URCap function that only works inside .urp programs.

    Required .urp files on the robot:
      - grip_close.urp  — structure: Loop > RG2(0) > Wait 0.1
                          Loops continuously, re-closing every 0.1s.
                          Stops only when a new program is loaded (via stop + load).
      - grip_open.urp   — structure: RG2(100)  (single action, no loop)

    Force is 40N in both programs (configured on pendant).

    SLIP DETECTION:
    Background thread monitors grip state. If the gripper was holding an object
    (DI8 HIGH + width > 12mm) and loses it (width drops to < 11mm), a slip is
    printed. The Loop in grip_close.urp handles re-closing automatically.

    TICKING FIX:
    grip_close.urp loops indefinitely. Before loading grip_open.urp, a 'stop'
    command is sent to terminate the loop. Without this, the open program loads
    while the close loop is still running, causing the gripper to tick.

    PORTS:
    - 29999: Dashboard Server  — stop / load / play .urp files
    - 30002: Secondary Client  — read robot state (width, force) ONLY
    """

    SLIP_CHECK_INTERVAL = 0.3    # seconds between slip checks in monitoring thread
    DASHBOARD_PORT      = 29999
    FEEDBACK_PORT       = 30002

    def __init__(self, ip):
        self.ip = ip

        # Configurable grip force shown in status (informational only —
        # actual force is set inside the .urp files on the robot)
        self.grip_force_note = "Set inside grip_close.urp on teach pendant"

        # I/O feedback state (updated by background thread)
        self.latest_digital_in  = 0
        self.latest_analog_in2  = 0.0
        self.running            = True

        # Grip state: "open" | "closing"
        self._grip_state      = "open"
        self._grip_state_lock = threading.Lock()

        # Slip / loop-stop detection state
        self._had_object        = False   # True once gripper confirmed holding object
        self._monitoring_active = False   # True after initial close settles (2.5s)
        self._loop_stopped      = False   # True once we've sent 'stop' after close

        # --- Connect feedback stream (port 30002) ---
        print(f"Connecting to UR10 at {self.ip}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3.0)
            self.sock.connect((self.ip, self.FEEDBACK_PORT))
            print(f"Connected to I/O feedback stream (port {self.FEEDBACK_PORT}).")
            self._recv_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._recv_thread.start()
        except Exception as e:
            print(f"Warning: Could not connect to I/O feedback: {e}")

        # --- Start slip detection monitor ---
        self._monitor_thread = threading.Thread(target=self._slip_monitor, daemon=True)
        self._monitor_thread.start()

    # ------------------------------------------------------------------ #
    #  I/O FEEDBACK (port 30002)                                          #
    # ------------------------------------------------------------------ #

    def _update_loop(self):
        """Background thread: parse real-time robot state from port 30002."""
        while self.running:
            try:
                header = self.sock.recv(4)
                if not header or len(header) < 4:
                    continue

                packet_len  = struct.unpack("!I", header)[0]
                packet_data = self.sock.recv(packet_len - 4)

                offset = 1  # skip message type byte
                while offset < len(packet_data):
                    if offset + 4 > len(packet_data):
                        break
                    p_size = struct.unpack("!I", packet_data[offset:offset+4])[0]
                    p_type = packet_data[offset+4]

                    if p_type == 2:  # Tool Data sub-package
                        # AI2 — gripper width voltage — 8-byte double at offset+7
                        ai2_bytes = packet_data[offset+7 : offset+15]
                        self.latest_analog_in2 = struct.unpack("!d", ai2_bytes)[0]
                        # DI8 — force limit bit — last byte of sub-package
                        self.latest_digital_in = packet_data[offset + p_size - 1]

                    offset += p_size

            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  DASHBOARD COMMANDS (port 29999)                                    #
    # ------------------------------------------------------------------ #

    def _dashboard_cmd(self, cmd: str) -> str:
        """Send one command to the Dashboard Server and return its response."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self.ip, self.DASHBOARD_PORT))
            sock.recv(1024)                          # consume welcome banner
            sock.sendall((cmd + "\n").encode())
            response = sock.recv(1024).decode().strip()
            sock.close()
            return response
        except Exception as e:
            print(f"Dashboard error [{cmd}]: {e}")
            return ""

    def _stop_and_wait(self, timeout: float = 4.0):
        """
        Send stop, poll until robot confirms not running, then wait an extra
        3 seconds to let the RG2 gripper's internal command buffer drain.
        The close loop fires at 10Hz so the buffer can have ~30 queued commands.
        """
        self._dashboard_cmd("stop")
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._dashboard_cmd("running")
            if "false" in status.lower():
                break
            time.sleep(0.15)
        print("  Waiting for gripper buffer to drain...")
        time.sleep(1.5)   # let RG2 internal queue flush (min safe value at 10Hz loop)

    def _play_urp(self, program_name: str):
        """Stop running program (confirmed + buffer drain), then load and play."""
        self._stop_and_wait()
        resp = self._dashboard_cmd(f"load {program_name}")
        if resp:
            time.sleep(0.2)
            self._dashboard_cmd("play")


    # ------------------------------------------------------------------ #
    #  SLIP DETECTION MONITOR                                             #
    # ------------------------------------------------------------------ #

    def _slip_monitor(self):
        """
        Background thread — runs during 'closing' state.

        Sequence:
          1. Wait for _monitoring_active (2.5s settle after start_closing)
          2. As soon as DI8 goes HIGH (jaws hit something), send 'stop' to
             kill the Loop. Gripper holds position quietly.
          3. Record whether an object is held (width > 12mm) or fully closed
             empty (width < 11mm).
          4. Keep watching. If object slips (was > 12mm, now < 11mm):
             - Print SLIP DETECTED
             - Restart grip_close.urp so it closes fully and stops again
        """
        while self.running:
            with self._grip_state_lock:
                state = self._grip_state

            if state == "closing" and self._monitoring_active:
                force = self.is_force_detected()
                width = self.get_width_mm()

                # Track if we have an object while loop is running
                if force and width > 12.0 and not self._had_object:
                    self._had_object = True
                    print(f"  Holding object at {width:.1f} mm — loop running.")

                # Only stop the loop when FULLY CLOSED (width < 11mm)
                if force and width < 11.0 and not self._loop_stopped:
                    if self._had_object:
                        # Was holding, now fully closed — object slipped!
                        print("\n[SLIP DETECTED] Object dropped — gripper closed fully.")
                    else:
                        print(f"  Fully closed ({width:.1f} mm) — stopping loop.")
                    self._dashboard_cmd("stop")
                    self._loop_stopped = True
                    self._had_object   = False

                # After a slip stop, re-enable monitoring for next slip
                # (start_closing resets _loop_stopped if called again)

            elif state != "closing":
                self._had_object        = False
                self._monitoring_active = False
                self._loop_stopped      = False

            time.sleep(self.SLIP_CHECK_INTERVAL)

    # ------------------------------------------------------------------ #
    #  PUBLIC GRIPPER API                                                 #
    # ------------------------------------------------------------------ #

    def start_closing(self):
        """
        Play grip_close.urp (Loop), then auto-stop it once the gripper
        contacts something (DI8 HIGH). Monitoring begins after 2.5s settle.
        """
        CLOSE_SETTLE_TIME = 2.5

        with self._grip_state_lock:
            self._grip_state = "closing"
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False

        print("Closing gripper...")
        self._play_urp("grip_close.urp")

        def _enable_monitoring():
            time.sleep(CLOSE_SETTLE_TIME)
            with self._grip_state_lock:
                if self._grip_state == "closing":
                    self._monitoring_active = True

        threading.Thread(target=_enable_monitoring, daemon=True).start()

    def start_opening(self):
        """Stop the close loop and open the gripper."""
        with self._grip_state_lock:
            self._grip_state = "open"
        self._had_object        = False
        self._monitoring_active = False
        self._loop_stopped      = False
        print("Opening gripper...")
        self._play_urp("grip_open.urp")   # _play_urp sends stop first

    @property
    def grip_state(self):
        with self._grip_state_lock:
            return self._grip_state

    # ------------------------------------------------------------------ #
    #  SENSOR READINGS                                                    #
    # ------------------------------------------------------------------ #

    def get_width_mm(self):
        """
        Calibrated gripper width (mm).

        Measured calibration points:
          Reading 65.8mm → Actual 91mm   (open)
          Reading  8.5mm → Actual 10.5mm (closed)

        Formula: actual = (raw * 1.405) - 1.44
        """
        voltage = max(self.latest_analog_in2, 0)
        raw_mm  = (voltage / 3.7) * 110.0
        slope   = (91.0 - 10.5) / (65.8 - 8.5)   # ≈ 1.405
        offset  = 10.5 - (8.5 * slope)             # ≈ -1.44
        return round((raw_mm * slope) + offset, 1)

    def is_force_detected(self):
        """True when DI8 HIGH — gripper jaws are touching something."""
        return (self.latest_digital_in & 0b00000001) != 0

    def is_object_detected(self):
        """
        True if an object is gripped (not just fully closed).

          DI8 HIGH + width > 12mm  → jaws on object     → OBJECT DETECTED
          DI8 HIGH + width < 11mm  → fully closed, empty → NO OBJECT
          DI8 LOW                  → open / moving        → NO OBJECT
        """
        if not self.is_force_detected():
            return False
        width = self.get_width_mm()
        return width > 12.0   # <11mm = empty close, 11-12mm = edge case treated as no object

    # ------------------------------------------------------------------ #
    #  CLEANUP                                                            #
    # ------------------------------------------------------------------ #

    def close(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()


# ====================================================================== #
#  MAIN — interactive test                                                #
# ====================================================================== #

if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"   # <- update to your robot's current IP

    try:
        controller = UR10Controller(ROBOT_IP)

        print("Waiting for I/O feedback to stabilise...")
        time.sleep(1.5)

        print("\n=== UR10 Gripper Control with Slip Detection ===")
        print("Commands:")
        print("  'c' = Close gripper  (slip detection active until you open)")
        print("  'o' = Open gripper   (exits slip detection mode)")
        print("  's' = Status")
        print("  'q' = Quit\n")
        print("NOTE: Grip force is configured inside grip_close.urp on the robot.\n")

        while True:
            cmd = input("Enter command (c/o/s/q): ").strip().lower()

            if cmd == 'q':
                break

            elif cmd == 'c':
                controller.start_closing()
                time.sleep(2.5)   # wait for gripper to finish moving

                width        = controller.get_width_mm()
                voltage      = controller.latest_analog_in2
                force_limit  = controller.is_force_detected()
                obj_detected = controller.is_object_detected()

                print(f"\n--- GRASP RESULT ---")
                print(f"Width:           {width} mm")
                print(f"Voltage (AI2):   {voltage:.3f} V")
                print(f"DI8 Force Limit: {force_limit}  (HIGH = jaws touching something)")

                if obj_detected:
                    print(f"\nOBJECT DETECTED  ({width} mm)")
                    print(f"  Slip detection active — will auto re-close if object drops.")
                else:
                    if force_limit and width < 11.0:
                        print(f"\nNO OBJECT — gripper fully closed ({width} mm)")
                    else:
                        print(f"\nNO OBJECT DETECTED")
                print(f"--------------------\n")

            elif cmd == 'o':
                controller.start_opening()
                time.sleep(2.5)
                width = controller.get_width_mm()
                print(f"Gripper open. Width: {width} mm\n")

            elif cmd == 's':
                width   = controller.get_width_mm()
                voltage = controller.latest_analog_in2
                state   = controller.grip_state

                print(f"\n--- Status ---")
                print(f"Grip Mode:           {state}")
                print(f"Slip Monitor Active: {controller._monitoring_active}  (enabled 2.5s after close)")
                print(f"Width:               {width} mm")
                print(f"Voltage (AI2):       {voltage:.3f} V")
                print(f"Grip Force:          40 N  (set in grip_close.urp / grip_open.urp on pendant)")
                print(f"DI8 Force Limit:     {controller.is_force_detected()}  (HIGH = jaws on something)")
                print(f"Object Detected:     {controller.is_object_detected()}")
                print(f"Had Object:          {controller._had_object}  (slip tracker)")
                print(f"Digital In Byte:     {bin(controller.latest_digital_in)}")
                print(f"--------------\n")

            else:
                print("Unknown command.\n")

        controller.close()
        print("Exiting.")

    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
