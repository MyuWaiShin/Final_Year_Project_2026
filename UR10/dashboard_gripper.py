import socket
import time

class UR10Dashboard:
    """
    Simple interface to UR10 Dashboard Server (port 29999) to trigger programs.
    """
    def __init__(self, ip):
        self.ip = ip
        self.port = 29999
        
    def send_command(self, cmd):
        """Send a command to the dashboard server and return response."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.ip, self.port))
            
            # Receive welcome message
            welcome = sock.recv(1024).decode()
            print(f"Dashboard: {welcome.strip()}")
            
            # Send command
            sock.sendall((cmd + "\n").encode())
            
            # Get response
            response = sock.recv(1024).decode()
            print(f"Response: {response.strip()}")
            
            sock.close()
            return response.strip()
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def load_program(self, program_name):
        """Load a .urp program file."""
        return self.send_command(f"load {program_name}")
    
    def play(self):
        """Start the loaded program."""
        return self.send_command("play")
    
    def stop(self):
        """Stop the current program."""
        return self.send_command("stop")

if __name__ == "__main__":
    ROBOT_IP = "192.168.8.102"
    
    dashboard = UR10Dashboard(ROBOT_IP)
    
    print("\n=== UR10 Gripper Control (Dashboard Mode) ===")
    print("Commands:")
    print("  'o' = Open Gripper")
    print("  'c' = Close Gripper")
    print("  'q' = Quit")
    print("\nUsing programs: grip_open.urp and grip_close.urp\n")
    
    while True:
        cmd = input("Enter command (o/c/q): ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'o':
            print("\n→ Opening gripper...")
            dashboard.load_program("grip_open.urp")
            time.sleep(0.5)
            dashboard.play()
            print("✓ Command sent. Gripper should open.\n")
        elif cmd == 'c':
            print("\n→ Closing gripper...")
            dashboard.load_program("grip_close.urp")
            time.sleep(0.5)
            dashboard.play()
            print("✓ Command sent. Gripper should close.\n")
        else:
            print("Unknown command.\n")
        
        time.sleep(0.5)
    
    print("Exiting.")
