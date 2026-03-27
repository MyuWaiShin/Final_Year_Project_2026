# save_calibration.py
"""Utility for saving and replaying eye‑in‑hand calibration poses.

Usage:
    python save_calibration.py capture   # capture current TCP pose and append to a JSON file
    python save_calibration.py replay   # replay all saved poses sequentially

The script assumes a `Robot` class with the following minimal API:
    robot = Robot()
    pose = robot.get_pose()               # returns a dict with keys ['x','y','z','rx','ry','rz']
    robot.move_to_pose(pose)              # moves robot to the given pose

The captured poses are stored in ``calibration_poses.json`` in the same directory.
"""
import json
import sys
from pathlib import Path

# TODO: replace this placeholder with your actual robot interface import
# from your_project.robot_interface import Robot

class Robot:
    """Placeholder Robot class – replace with the real implementation that
    communicates with the UR10 and reads the current TCP pose.
    """
    def get_pose(self):
        # Return a dummy pose; replace with real sensor reading
        return {"x": 0.0, "y": 0.0, "z": 0.2, "rx": 0.0, "ry": 0.0, "rz": 0.0}

    def move_to_pose(self, pose):
        # Implement actual motion command here
        print(f"Moving to pose: {pose}")

def _load_poses(file_path: Path):
    if not file_path.exists():
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_poses(file_path: Path, poses):
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)

def capture_pose():
    robot = Robot()
    pose = robot.get_pose()
    file_path = Path(__file__).with_name("calibration_poses.json")
    poses = _load_poses(file_path)
    poses.append(pose)
    _save_poses(file_path, poses)
    print(f"Captured pose saved to {file_path}. Total poses: {len(poses)}")

def replay_poses():
    robot = Robot()
    file_path = Path(__file__).with_name("calibration_poses.json")
    poses = _load_poses(file_path)
    if not poses:
        print("No saved poses found.")
        return
    for idx, pose in enumerate(poses, start=1):
        print(f"Replaying pose {idx}/{len(poses)}: {pose}")
        robot.move_to_pose(pose)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"capture", "replay"}:
        print("Usage: python save_calibration.py [capture|replay]")
        sys.exit(1)
    if sys.argv[1] == "capture":
        capture_pose()
    else:
        replay_poses()
