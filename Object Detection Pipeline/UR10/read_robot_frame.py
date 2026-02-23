import socket
import struct
import time

class UR10TCPReader:
    """
    Read TCP (Tool Center Point) position from UR10 robot.
    Connects to port 30002 (Secondary Client) and parses Cartesian Info packets.
    
    TCP Position format: [X, Y, Z, Rx, Ry, Rz]
    - X, Y, Z: Position in meters
    - Rx, Ry, Rz: Rotation vector (axis-angle representation)
    """
    
    def __init__(self, ip):
        self.ip = ip
        self.port = 30002
        
        # Latest TCP position
        self.tcp_x = 0.0
        self.tcp_y = 0.0
        self.tcp_z = 0.0
        self.tcp_rx = 0.0
        self.tcp_ry = 0.0
        self.tcp_rz = 0.0
        
        print(f"Connecting to UR10 at {self.ip}:{self.port}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.ip, self.port))
            print("Connected successfully.")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise
    
    def read_tcp_position(self):
        """
        Read one packet and extract TCP position.
        Returns: (x, y, z, rx, ry, rz) or None if packet not found.
        """
        try:
            # Read packet header (4 bytes size)
            data = self.sock.recv(4)
            if not data or len(data) < 4:
                return None
                
            packet_len = struct.unpack("!I", data)[0]
            packet_data = self.sock.recv(packet_len - 4)
            
            # Parse sub-packages
            offset = 1  # Skip message type
            
            while offset < len(packet_data):
                if offset + 4 > len(packet_data):
                    break
                    
                p_size = struct.unpack("!I", packet_data[offset:offset+4])[0]
                p_type = packet_data[offset+4]
                
                # Package Type 4 = Cartesian Info (contains TCP position)
                if p_type == 4:
                    # Parse TCP position (X, Y, Z, Rx, Ry, Rz)
                    self.tcp_x = struct.unpack("!d", packet_data[offset+5:offset+13])[0]
                    self.tcp_y = struct.unpack("!d", packet_data[offset+13:offset+21])[0]
                    self.tcp_z = struct.unpack("!d", packet_data[offset+21:offset+29])[0]
                    self.tcp_rx = struct.unpack("!d", packet_data[offset+29:offset+37])[0]
                    self.tcp_ry = struct.unpack("!d", packet_data[offset+37:offset+45])[0]
                    self.tcp_rz = struct.unpack("!d", packet_data[offset+45:offset+53])[0]
                    
                    return (self.tcp_x, self.tcp_y, self.tcp_z, 
                            self.tcp_rx, self.tcp_ry, self.tcp_rz)
                
                offset += p_size
                
        except Exception as e:
            print(f"Error reading TCP: {e}")
            return None
    
    def get_tcp_position(self):
        """Return the latest TCP position as a dictionary."""
        return {
            'x': self.tcp_x * 1000,  # Convert to mm
            'y': self.tcp_y * 1000,
            'z': self.tcp_z * 1000,
            'rx': self.tcp_rx,
            'ry': self.tcp_ry,
            'rz': self.tcp_rz
        }
    
    def close(self):
        """Close the connection."""
        if hasattr(self, 'sock'):
            self.sock.close()


if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"
    
    try:
        reader = UR10TCPReader(ROBOT_IP)
        
        print("\n" + "="*60)
        print("  UR10 TCP Position Reader")
        print("="*60)
        print("Press Ctrl+C to stop\n")
        
        while True:
            # Read TCP position
            tcp = reader.read_tcp_position()
            
            if tcp:
                pos = reader.get_tcp_position()
                
                # Clear line and print formatted output
                print(f"\rPosition (mm):  X={pos['x']:8.1f}  Y={pos['y']:8.1f}  Z={pos['z']:8.1f}  |  "
                      f"Rotation (rad):  Rx={pos['rx']:6.3f}  Ry={pos['ry']:6.3f}  Rz={pos['rz']:6.3f}", 
                      end='', flush=True)
            
            time.sleep(0.1)  # Update at ~10Hz
        
    except KeyboardInterrupt:
        print("\n\nStopped.")
        reader.close()
    except Exception as e:
        print(f"\nError: {e}")
