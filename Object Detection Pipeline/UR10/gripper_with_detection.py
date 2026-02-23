import socket
import struct
import time
import threading

class UR10Controller:
    """
    Combined Dashboard + I/O Feedback controller for UR10 with RG2 Gripper.
    - Dashboard (port 29999): Load and play programs
    - Secondary Client (port 30002): Read real-time I/O feedback
    """
    
    def __init__(self, ip):
        self.ip = ip
        self.dashboard_port = 29999
        self.secondary_port = 30002
        
        # Feedback state
        self.latest_digital_in = 0
        self.latest_analog_in2 = 0.0
        self.running = True
        
        # Connect to secondary client for feedback
        print(f"Connecting to UR10 at {self.ip} for I/O feedback...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.ip, self.secondary_port))
            print("Connected to I/O feedback stream.")
            
            # Start background thread to read feedback
            self.recv_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.recv_thread.start()
            
        except Exception as e:
            print(f"Warning: Could not connect to I/O feedback: {e}")
    
    def _update_loop(self):
        """Background loop to parse I/O data from port 30002."""
        while self.running:
            try:
                # Read packet header
                data = self.sock.recv(4)
                if not data or len(data) < 4:
                    continue
                    
                packet_len = struct.unpack("!I", data)[0]
                packet_data = self.sock.recv(packet_len - 4)
                
                # Parse sub-packages
                offset = 1  # Skip message type
                
                while offset < len(packet_data):
                    if offset + 4 > len(packet_data):
                        break
                    p_size = struct.unpack("!I", packet_data[offset:offset+4])[0]
                    p_type = packet_data[offset+4]
                    
                    if p_type == 2:  # Tool Data
                        # AI2 (Analog Input 2) at offset+7 to offset+15 (8 bytes double)
                        ai2_bytes = packet_data[offset+7 : offset+15]
                        self.latest_analog_in2 = struct.unpack("!d", ai2_bytes)[0]
                        
                        # Digital inputs at last byte of package
                        digital_byte = packet_data[offset + p_size - 1]
                        self.latest_digital_in = digital_byte
                        
                    offset += p_size
                    
            except Exception:
                pass
    
    def send_dashboard_command(self, cmd):
        """Send command to Dashboard Server (port 29999)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.ip, self.dashboard_port))
            
            # Receive welcome
            sock.recv(1024)
            
            # Send command
            sock.sendall((cmd + "\n").encode())
            
            # Get response
            response = sock.recv(1024).decode().strip()
            sock.close()
            return response
            
        except Exception as e:
            print(f"Dashboard error: {e}")
            return None
    
    def load_program(self, program_name):
        """Load a .urp program."""
        return self.send_dashboard_command(f"load {program_name}")
    
    def play(self):
        """Start the loaded program."""
        return self.send_dashboard_command("play")
    
    def get_width_mm(self):
        """
        Calculate gripper width using LINEAR CALIBRATION from actual measurements.
        
        Measured data points:
        - Reading: 65.8mm → Actual: 91mm (open)
        - Reading: 8.5mm → Actual: 10.5mm (closed)
        
        Linear interpolation:
        slope = (91 - 10.5) / (65.8 - 8.5) = 80.5 / 57.3 ≈ 1.405
        offset = 10.5 - (8.5 * 1.405) ≈ -1.44
        
        Formula: actual_width = (raw_width * 1.405) - 1.44
        """
        voltage = self.latest_analog_in2
        if voltage < 0:
            voltage = 0
        
        # First convert voltage to raw width (using datasheet formula)
        raw_width_mm = (voltage / 3.7) * 110.0
        
        # Then apply calibration
        # Using your measurements: 65.8 → 91, 8.5 → 10.5
        slope = (91.0 - 10.5) / (65.8 - 8.5)  # ≈ 1.405
        offset = 10.5 - (8.5 * slope)  # ≈ -1.44
        
        actual_width = (raw_width_mm * slope) + offset
        
        return round(actual_width, 1)
    
    def is_force_detected(self):
        """
        Check if DI8 (force sensor) is high.
        DI8 goes HIGH when gripper jaws make contact (either with object or each other).
        """
        bit0 = (self.latest_digital_in & 0b00000001) != 0
        return bit0
    
    def is_object_detected(self):
        """
        Detect if an object is grasped using COMBINED DI8 + WIDTH detection.
        
        Logic:
        - DI8 HIGH + width < 11mm  → Jaws touching each other (NO object)
        - DI8 HIGH + width > 12mm  → Jaws touching object (OBJECT detected)
        - DI8 LOW                  → Gripper still moving or open (NO object)
        
        Based on actual measurements:
        - Fully closed (no object): 8.5-10.5mm
        - Objects detected: 15mm+
        
        This distinguishes between "force from closing fully" vs "force from gripping object".
        """
        force_detected = self.is_force_detected()
        width = self.get_width_mm()
        
        # If no force detected at all, no object
        if not force_detected:
            return False
        
        # Force detected - check width to distinguish
        if width < 11.0:
            # Fully closed, jaws touching each other
            return False
        elif width > 12.0:
            # Object detected between jaws
            return True
        else:
            # Edge case: 11-12mm range, assume no object to be safe
            return False
    
    def get_debug_info(self):
        """Get detailed debug information about digital inputs."""
        byte_val = self.latest_digital_in
        return {
            'raw_byte': byte_val,
            'binary': format(byte_val, '08b'),
            'bit0_DI8': (byte_val & 0b00000001) != 0,
            'bit1_DI9': (byte_val & 0b00000010) != 0,
            'bit2': (byte_val & 0b00000100) != 0,
            'bit3': (byte_val & 0b00001000) != 0,
            'bit4': (byte_val & 0b00010000) != 0,
            'bit5': (byte_val & 0b00100000) != 0,
            'bit6': (byte_val & 0b01000000) != 0,
            'bit7': (byte_val & 0b10000000) != 0,
        }
    
    def close(self):
        """Clean up connections."""
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()


if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"
    
    try:
        controller = UR10Controller(ROBOT_IP)
        
        # Wait for initial feedback
        print("Waiting for I/O feedback...")
        time.sleep(1.5)
        
        print("\n=== UR10 Gripper Control with Grasp Detection ===")
        print("Commands:")
        print("  'o' = Open Gripper")
        print("  'c' = Close Gripper (with object detection)")
        print("  's' = Show current status")
        print("  'q' = Quit\n")
        
        while True:
            cmd = input("Enter command (o/c/s/q): ").strip().lower()
            
            if cmd == 'q':
                break
                
            elif cmd == 'o':
                print("\n→ Opening gripper...")
                controller.load_program("grip_open.urp")
                time.sleep(0.3)
                controller.play()
                time.sleep(2.5)  # Wait for movement
                
                width = controller.get_width_mm()
                print(f"✓ Gripper opened. Width: {width} mm\n")
                
            elif cmd == 'c':
                print("\n→ Closing gripper...")
                controller.load_program("grip_close.urp")
                time.sleep(0.3)
                controller.play()
                time.sleep(2.5)  # Wait for movement
                
                # Get info
                width = controller.get_width_mm()
                voltage = controller.latest_analog_in2
                force_detected = controller.is_force_detected()
                object_detected = controller.is_object_detected()
                
                print(f"\n--- GRASP RESULT ---")
                print(f"Width: {width} mm")
                print(f"Voltage: {voltage:.3f}V")
                print(f"DI8 (Force): {force_detected}")
                
                # Detection result
                if object_detected:
                    print(f"\n✓ OBJECT DETECTED in gripper!")
                    print(f"  Gripper width: {width} mm")
                else:
                    if force_detected and width < 9.0:
                        print(f"\n✗ NO OBJECT DETECTED (gripper fully closed)")
                        print(f"  Gripper width: {width} mm")
                    else:
                        print(f"\n✗ NO OBJECT DETECTED")
                        print(f"  Gripper width: {width} mm")
                print(f"--------------------\n")
                
            elif cmd == 's':
                # Show current status
                width = controller.get_width_mm()
                force = controller.is_force_detected()
                voltage = controller.latest_analog_in2
                
                print(f"\n--- Current Status ---")
                print(f"Width: {width} mm")
                print(f"Voltage (AI2): {voltage:.2f} V")
                print(f"Force Detected (DI8): {force}")
                print(f"Digital In Byte: {bin(controller.latest_digital_in)}")
                print(f"----------------------\n")
                
            else:
                print("Unknown command.\n")
        
        controller.close()
        print("Exiting.")
        
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}")
