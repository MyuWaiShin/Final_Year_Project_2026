import socket
import struct
import time
import threading

class VoltageChecker:
    """
    Simple utility to check actual analog voltage from RG2 gripper.
    Displays raw voltage and calculated width to verify configuration.
    """
    
    def __init__(self, ip):
        self.ip = ip
        self.port = 30002
        self.latest_analog_in2 = 0.0
        self.running = True
        
        print(f"Connecting to UR10 at {self.ip}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(2.0)
        self.sock.connect((self.ip, self.port))
        print("Connected.\n")
        
        # Start background thread
        self.recv_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.recv_thread.start()
    
    def _update_loop(self):
        """Background loop to read AI2 voltage."""
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
                    
                    # Tool Data (AI2)
                    if p_type == 2:
                        ai2_bytes = packet_data[offset+7:offset+15]
                        self.latest_analog_in2 = struct.unpack("!d", ai2_bytes)[0]
                    
                    offset += p_size
                    
            except Exception:
                pass
    
    def get_width_5v(self):
        """Calculate width assuming 5V configuration (max 3.7V)."""
        voltage = self.latest_analog_in2
        raw_width = (voltage / 3.7) * 110.0
        
        # Apply your calibration
        slope = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        actual_width = (raw_width * slope) + offset
        
        return round(actual_width, 1)
    
    def get_width_10v(self):
        """Calculate width assuming 10V configuration (max 3.0V)."""
        voltage = self.latest_analog_in2
        raw_width = (voltage / 3.0) * 110.0
        
        # Apply your calibration
        slope = (91.0 - 10.5) / (65.8 - 8.5)
        offset = 10.5 - (8.5 * slope)
        actual_width = (raw_width * slope) + offset
        
        return round(actual_width, 1)
    
    def close(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()


if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"
    
    try:
        checker = VoltageChecker(ROBOT_IP)
        
        # Wait for initial data
        time.sleep(1.5)
        
        print("=" * 70)
        print("  RG2 Gripper Voltage Checker")
        print("=" * 70)
        print("\nThis will help you determine if you're using 5V or 10V configuration.")
        print("\nExpected voltage ranges:")
        print("  - 5V mode (0V:5V):  0.0V to ~3.7V")
        print("  - 10V mode (0V:10V): 0.0V to ~3.0V")
        print("\nPress Ctrl+C to stop\n")
        print("-" * 70)
        
        while True:
            voltage = checker.latest_analog_in2
            width_5v = checker.get_width_5v()
            width_10v = checker.get_width_10v()
            
            print(f"\rRaw Voltage: {voltage:.4f}V  |  "
                  f"Width (5V mode): {width_5v:5.1f}mm  |  "
                  f"Width (10V mode): {width_10v:5.1f}mm", 
                  end='', flush=True)
            
            time.sleep(0.2)
        
    except KeyboardInterrupt:
        print("\n\nStopped.")
        checker.close()
    except Exception as e:
        print(f"\nError: {e}")
