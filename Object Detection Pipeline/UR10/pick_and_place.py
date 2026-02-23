import socket
import struct
import time
import threading

class UR10PickAndPlace:
    """
    Pick and place controller for UR10 with RG2 gripper.
    Combines dashboard control, gripper detection, and position monitoring.
    """
    
    def __init__(self, ip):
        self.ip = ip
        self.dashboard_port = 29999
        self.secondary_port = 30002
        
        # Gripper feedback state
        self.latest_digital_in = 0
        self.latest_analog_in2 = 0.0
        self.running = True
        
        # TCP position
        self.tcp_x = 0.0
        self.tcp_y = 0.0
        self.tcp_z = 0.0
        
        print(f"Connecting to UR10 at {self.ip}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.ip, self.secondary_port))
            print("Connected to robot feedback stream.")
            
            # Start background thread to read feedback
            self.recv_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.recv_thread.start()
            
        except Exception as e:
            print(f"Connection error: {e}")
            raise
    
    def _update_loop(self):
        """Background loop to parse I/O and position data."""
        while self.running:
            try:
                data = self.sock.recv(4)
                if not data or len(data) < 4:
                    continue
                    
                packet_len = struct.unpack("!I", data)[0]
                packet_data = self.sock.recv(packet_len - 4)
                
                offset = 1
                while offset < len(packet_data):
                    if offset + 4 > len(packet_data):
                        break
                    p_size = struct.unpack("!I", packet_data[offset:offset+4])[0]
                    p_type = packet_data[offset+4]
                    
                    # Tool Data (gripper feedback)
                    if p_type == 2:
                        ai2_bytes = packet_data[offset+7:offset+15]
                        self.latest_analog_in2 = struct.unpack("!d", ai2_bytes)[0]
                        digital_byte = packet_data[offset + p_size - 1]
                        self.latest_digital_in = digital_byte
                    
                    # Cartesian Info (TCP position)
                    elif p_type == 4:
                        self.tcp_x = struct.unpack("!d", packet_data[offset+5:offset+13])[0]
                        self.tcp_y = struct.unpack("!d", packet_data[offset+13:offset+21])[0]
                        self.tcp_z = struct.unpack("!d", packet_data[offset+21:offset+29])[0]
                    
                    offset += p_size
                    
            except Exception:
                pass
    
    def send_dashboard_command(self, cmd):
        """Send command to Dashboard Server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.ip, self.dashboard_port))
            sock.recv(1024)  # Welcome message
            sock.sendall((cmd + "\n").encode())
            response = sock.recv(1024).decode().strip()
            sock.close()
            return response
        except Exception as e:
            print(f"Dashboard error: {e}")
            return None
    
    def move_to(self, x, y, z, rx, ry, rz, acc=0.5, vel=0.5, fast=False):
        """
        Move to position using URScript movel command (linear movement).
        x, y, z in meters, rx, ry, rz in radians.
        fast: if True, use shorter wait time for approach/retract movements
        """
        script = f"""def move_program():
  movel(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={acc}, v={vel})
end
"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, 30002))
        sock.send(script.encode())
        sock.close()
        
        # Wait for movement to complete
        wait_time = 3.5 if fast else 5.0
        time.sleep(wait_time)
    
    def move_to_j(self, x, y, z, rx, ry, rz, acc=0.5, vel=0.5):
        """
        Move to position using URScript movej command (joint movement).
        Better for long-distance moves, avoids singularities.
        x, y, z in meters, rx, ry, rz in radians.
        """
        script = f"""def move_program():
  movej(p[{x}, {y}, {z}, {rx}, {ry}, {rz}], a={acc}, v={vel})
end
"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, 30002))
        sock.send(script.encode())
        sock.close()
        
        # Wait for movement to complete (longer for joint moves)
        time.sleep(6.0)
    
    def open_gripper(self):
        """Open gripper using dashboard."""
        self.send_dashboard_command("load grip_open.urp")
        time.sleep(0.3)
        self.send_dashboard_command("play")
        time.sleep(2.5)
    
    def close_gripper(self):
        """Close gripper using dashboard."""
        self.send_dashboard_command("load grip_close.urp")
        time.sleep(0.3)
        self.send_dashboard_command("play")
        time.sleep(2.5)
    
    def get_width_mm(self):
        """Calculate gripper width from AI2 voltage with calibration."""
        voltage = self.latest_analog_in2
        if voltage < 0:
            voltage = 0
        raw_width_mm = (voltage / 3.7) * 110.0
        slope = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        actual_width = (raw_width_mm * slope) + offset
        return round(actual_width, 1)
    
    def is_object_detected(self):
        """Detect object using DI8 + width."""
        force_detected = (self.latest_digital_in & 1) != 0
        width = self.get_width_mm()
        
        if not force_detected:
            return False
        if width < 11.0:
            return False
        elif width > 12.0:
            return True
        else:
            return False
    
    def monitor_grip_during_transfer(self, duration_sec=6.0, check_interval=0.5):
        """
        Monitor grip continuously during transfer movement.
        Returns True if object is still held, False if dropped.
        """
        start_time = time.time()
        while (time.time() - start_time) < duration_sec:
            if not self.is_object_detected():
                width = self.get_width_mm()
                print(f"\n⚠️  Object dropped during transfer! Width: {width} mm")
                return False
            time.sleep(check_interval)
        return True
    
    def pick_and_place(self, pick_x, pick_y, pick_z, place_x, place_y, place_z, 
                       rx, ry, rz, approach_offset=0.05, vel=0.5, acc=0.5):
        """
        Execute pick and place sequence.
        Positions in meters, approach_offset in meters (default 50mm).
        vel: velocity in m/s, acc: acceleration in m/s²
        """
        print(f"\n→ Moving to pick approach position...")
        self.move_to(pick_x, pick_y, pick_z + approach_offset, rx, ry, rz, acc=acc, vel=vel, fast=True)
        
        print("→ Opening gripper...")
        self.open_gripper()
        
        print("→ Moving down to pick...")
        self.move_to(pick_x, pick_y, pick_z, rx, ry, rz, acc=acc, vel=vel)
        
        print("→ Closing gripper...")
        self.close_gripper()
        
        # Check if object was grasped
        if self.is_object_detected():
            width = self.get_width_mm()
            print(f"✓ Object grasped! Width: {width} mm")
        else:
            width = self.get_width_mm()
            print(f"✗ No object detected! Width: {width} mm")
            print("  Aborting this pick...")
            self.move_to(pick_x, pick_y, pick_z + approach_offset, rx, ry, rz, acc=acc, vel=vel, fast=True)
            return False
        
        print("→ Lifting object...")
        # Start monitoring in parallel with lift movement
        lift_thread = threading.Thread(target=lambda: self.move_to(pick_x, pick_y, pick_z + approach_offset, rx, ry, rz, acc=acc, vel=vel, fast=True))
        lift_thread.start()
        
        # Monitor grip during lift
        if not self.monitor_grip_during_transfer(duration_sec=3.5):
            lift_thread.join()
            print("✗ Object lost during lift. Aborting...")
            return False
        lift_thread.join()
        
        print("→ Moving to place approach position...")
        # Start monitoring in parallel with transfer movement
        transfer_thread = threading.Thread(target=lambda: self.move_to_j(place_x, place_y, place_z + approach_offset, rx, ry, rz, acc=acc, vel=vel))
        transfer_thread.start()
        
        # Monitor grip during transfer
        if not self.monitor_grip_during_transfer(duration_sec=6.0):
            transfer_thread.join()
            print("✗ Object lost during transfer. Aborting...")
            # Move to safe position (current place approach)
            time.sleep(1.0)
            return False
        transfer_thread.join()
        
        print("→ Moving down to place...")
        self.move_to(place_x, place_y, place_z, rx, ry, rz, acc=acc, vel=vel)
        
        print("→ Opening gripper...")
        self.open_gripper()
        
        print("→ Retracting...")
        self.move_to(place_x, place_y, place_z + approach_offset, rx, ry, rz, acc=acc, vel=vel, fast=True)
        
        print("✓ Pick and place complete!\n")
        return True
    
    def close(self):
        """Clean up connections."""
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()


if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"
    
    # Pick positions (3 cubes, 200mm apart in X) - CORRECTED
    PICK_POSITIONS = [
        (-0.2606, -1.1581, -0.4592),  # Cube 1
        (-0.0606, -1.1581, -0.4592),  # Cube 2 (X + 200mm)
        (0.1394, -1.1581, -0.4592),   # Cube 3 (X + 400mm)
    ]
    
    # Place position (same for all) - CORRECTED
    PLACE_POSITION = (0.7952, -0.8659, -0.3915)
    
    # Rotation - CORRECTED (using pick rotation)
    ROTATION = (2.987, 0.136, -0.005)
    
    # Home position (safe starting position)
    HOME_POSITION = (-0.0645, -0.7940, 1.0128)  # X, Y, Z in meters
    HOME_ROTATION = (1.709, -1.750, 0.709)       # Rx, Ry, Rz in radians
    
    # Movement parameters (adjust these to change speed)
    VELOCITY = 0.5      # m/s (0.1 = slow, 0.5 = medium, 1.0 = fast)
    ACCELERATION = 0.5  # m/s² (0.3 = slow, 0.8 = medium, 1.2 = fast)
    
    try:
        controller = UR10PickAndPlace(ROBOT_IP)
        
        # Wait for initial feedback
        print("Waiting for robot feedback...")
        time.sleep(1.5)
        
        print("\n" + "="*60)
        print("  UR10 Pick and Place - 3 Cubes")
        print("="*60)
        print(f"Pick positions: {len(PICK_POSITIONS)} cubes")
        print(f"Place position: X={PLACE_POSITION[0]*1000:.1f}, Y={PLACE_POSITION[1]*1000:.1f}, Z={PLACE_POSITION[2]*1000:.1f} mm")
        print(f"Home position: X={HOME_POSITION[0]*1000:.1f}, Y={HOME_POSITION[1]*1000:.1f}, Z={HOME_POSITION[2]*1000:.1f} mm")
        print(f"Approach offset: 100mm")
        print("="*60 + "\n")
        
        input("Press Enter to start pick and place sequence...")
        
        # Move to home position first
        print("\n→ Moving to home position...")
        controller.move_to_j(HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2],
                           HOME_ROTATION[0], HOME_ROTATION[1], HOME_ROTATION[2],
                           acc=ACCELERATION, vel=VELOCITY)
        print("✓ At home position")
        
        # Open gripper for safety at start
        print("→ Opening gripper...")
        controller.open_gripper()
        print("✓ Gripper open\n")
        time.sleep(0.5)
        
        for i, pick_pos in enumerate(PICK_POSITIONS, 1):
            print(f"\n{'='*60}")
            print(f"  CUBE {i} of {len(PICK_POSITIONS)}")
            print(f"{'='*60}")
            print(f"Pick: X={pick_pos[0]*1000:.1f}, Y={pick_pos[1]*1000:.1f}, Z={pick_pos[2]*1000:.1f} mm")
            
            success = controller.pick_and_place(
                pick_pos[0], pick_pos[1], pick_pos[2],
                PLACE_POSITION[0], PLACE_POSITION[1], PLACE_POSITION[2],
                ROTATION[0], ROTATION[1], ROTATION[2],
                approach_offset=0.100,  # 100mm
                vel=VELOCITY,
                acc=ACCELERATION
            )
            
            if not success:
                print(f"Failed to pick cube {i}. Continuing to next cube...")
            
            time.sleep(1.0)
        
        print("\n" + "="*60)
        print("  ALL CUBES PROCESSED")
        print("="*60)
        
        # Return to home position
        print("\n→ Returning to home position...")
        controller.move_to_j(HOME_POSITION[0], HOME_POSITION[1], HOME_POSITION[2],
                           HOME_ROTATION[0], HOME_ROTATION[1], HOME_ROTATION[2],
                           acc=ACCELERATION, vel=VELOCITY)
        print("✓ At home position")
        
        # Close gripper for safety after returning home
        print("\n→ Closing gripper for safety...")
        controller.close_gripper()
        print("✓ Gripper closed")
        
        controller.close()
        
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        controller.close()
    except Exception as e:
        print(f"\nError: {e}")
