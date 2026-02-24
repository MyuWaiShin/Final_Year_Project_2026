import socket
import struct
import numpy as np

class UR10PoseReader:
    """
    Reads real-time Cartesian pose from UR10 (CB3) over port 30003.
    """
    def __init__(self, ip="192.168.8.102"):
        self.ip = ip
        self.port = 30003
        
    def get_actual_tcp_pose(self):
        """
        Returns [X, Y, Z, Rx, Ry, Rz] in meters and radians.
        CB3 Real-time packet is 1060 bytes.
        TCP pose starts at byte 444 (6 doubles).
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect((self.ip, self.port))
                data = s.recv(1100) # Read one packet
                
                if len(data) < 444 + 48:
                    return None
                
                # Unpack 6 doubles starting at offset 444
                pose_data = data[444 : 444 + 48]
                pose = struct.unpack("!6d", pose_data)
                return list(pose)
        except Exception as e:
            print(f"[UR10] Error reading pose: {e}")
            return None

def rotvec_to_matrix(rx, ry, rz):
    """Converts UR rotation vector (axis-angle) to 3x3 rotation matrix."""
    theta = np.sqrt(rx**2 + ry**2 + rz**2)
    if theta < 1e-9:
        return np.eye(3)
    
    ux, uy, uz = rx/theta, ry/theta, rz/theta
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    
    R = np.array([
        [ux*ux*v + c,    ux*uy*v - uz*s, ux*uz*v + uy*s],
        [ux*uy*v + uz*s, uy*uy*v + c,    uy*uz*v - ux*s],
        [ux*uz*v - uy*s, uy*uz*v + ux*s, uz*uz*v + c   ]
    ])
    return R

def pose_to_matrix(pose):
    """Converts [X, Y, Z, Rx, Ry, Rz] to 4x4 transformation matrix."""
    x, y, z, rx, ry, rz = pose
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T

def matrix_to_pose(T):
    """Converts 4x4 matrix to UR pose [X, Y, Z, Rx, Ry, Rz]."""
    x, y, z = T[:3, 3]
    R = T[:3, :3]
    
    # Rotation matrix to axis-angle (UR format)
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle < 1e-9:
        return [x, y, z, 0, 0, 0]
    
    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(angle))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(angle))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(angle))
    
    return [x, y, z, rx*angle, ry*angle, rz*angle]
