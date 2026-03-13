"""
solve_calibration.py — Eye-in-Hand Calibration: Offline Solver
==============================================================
Loads the images and robot poses saved by collect_poses.py, runs
cv2.calibrateHandEye() with multiple methods, picks the best result
by reprojection error, and saves handeye_calibration.json.

Run:
    python solve_calibration.py

Output:
    handeye_calibration.json
"""

import json
import time
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
DATA_DIR   = Path("data")
# Can be overridden by replay_positions.py via env var
POSES_FILE = Path(_os.environ.get("HANDEYE_POSES_FILE", DATA_DIR / "poses.json"))
OUT_FILE   = Path("handeye_calibration.json")

METHODS = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility: rotation vector ↔ matrix
# ─────────────────────────────────────────────────────────────────────────────
def rotvec_to_matrix(rx, ry, rz):
    """UR axis-angle (Rodrigues) → 3×3 rotation matrix."""
    vec   = np.array([rx, ry, rz], dtype=np.float64)
    angle = np.linalg.norm(vec)
    if angle < 1e-9:
        return np.eye(3)
    axis = vec / angle
    K = np.array([[    0,    -axis[2],  axis[1]],
                  [ axis[2],     0,    -axis[0]],
                  [-axis[1],  axis[0],     0   ]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_rotvec(R):
    """3×3 → OpenCV-style rodrigues vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


# ─────────────────────────────────────────────────────────────────────────────
# Build 3-D object points for the checkerboard
# ─────────────────────────────────────────────────────────────────────────────
def make_obj_points(cols, rows, square_m):
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= square_m
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Compute per-sample reprojection error
# ─────────────────────────────────────────────────────────────────────────────
def reprojection_errors(obj_pts_list, img_pts_list, rvecs, tvecs,
                        camera_matrix, dist_coeffs):
    errs = []
    for obj, img, rv, tv in zip(obj_pts_list, img_pts_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, camera_matrix, dist_coeffs)
        err = np.linalg.norm(img.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        errs.append(err.mean())  # pixels
    return errs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a hand-eye result: mean reprojection error of board via H_cam2tcp
# ─────────────────────────────────────────────────────────────────────────────
def eval_handeye(R_c2tcp, t_c2tcp,
                 R_g2b_list, t_g2b_list,
                 rvecs_b2c, tvecs_b2c):
    """
    Evaluate hand-eye consistency by projecting the board to the robot base frame.
    Calculates the 3D position of the board origin in the robot base for each sample.
    Returns (mean_stdev_meters, mean_position_base).
    """
    board_positions_base = []
    
    # H_cam2tcp: Camera to TCP
    T_c2t = np.eye(4)
    T_c2t[:3, :3] = R_c2tcp
    T_c2t[:3, 3] = t_c2tcp.ravel()
    
    for R_g2b, t_g2b, rv_b2c, tv_b2c in zip(R_g2b_list, t_g2b_list, rvecs_b2c, tvecs_b2c):
        # 1. H_board2cam: Board to Camera
        R_b2c, _ = cv2.Rodrigues(rv_b2c)
        T_b2c = np.eye(4)
        T_b2c[:3, :3] = R_b2c
        T_b2c[:3, 3] = tv_b2c.ravel()
        
        # 2. H_tcp2base: TCP to Base
        T_t2b = np.eye(4)
        T_t2b[:3, :3] = R_g2b
        T_t2b[:3, 3] = t_g2b.ravel()
        
        # 3. H_board2base = H_tcp2base * H_cam2tcp * H_board2cam
        # Note: calibrateHandEye returns R_cam2gripper and t_cam2gripper
        # But UR10 TCP pose is Base -> Gripper.
        T_b2b = T_t2b @ T_c2t @ T_b2c
        
        # Extract board origin in base frame
        board_positions_base.append(T_b2b[:3, 3])
        
    board_positions_base = np.array(board_positions_base)
    std = np.std(board_positions_base, axis=0) # [std_x, std_y, std_z]
    mean_std = np.linalg.norm(std)
    mean_pos = np.mean(board_positions_base, axis=0)
    
    return float(mean_std), mean_pos


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not POSES_FILE.exists():
        print(f"ERROR: {POSES_FILE} not found -- run collect_poses.py first.")
        return

    with open(POSES_FILE) as f:
        data = json.load(f)

    board_type = data["board"].get("type", "checkerboard")
    if board_type != "charuco":
        print(f"ERROR: Poses file contains {board_type} data, but solver expects charuco.")
        return

    board_cols = data["board"]["cols"]
    board_rows = data["board"]["rows"]
    square_m   = data["board"]["square_m"]
    marker_m   = data["board"]["marker_m"]
    camera_matrix = np.array(data["camera"]["matrix"],  dtype=np.float64)
    dist_coeffs   = np.array(data["camera"]["distortion"], dtype=np.float64)
    samples    = data["samples"]

    print(f"\n=== Eye-in-Hand Calibration Solver ===")
    print(f"  Board: ChArUco {board_cols}x{board_rows}, square={square_m*1000:.0f}mm, marker={marker_m*1000:.0f}mm")
    print(f"  Samples: {len(samples)}")

    # Initialize Charuco Board
    aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    charuco_brD  = aruco.CharucoBoard((board_cols, board_rows), square_m, marker_m, aruco_dict)
    charuco_detector = aruco.CharucoDetector(charuco_brD)

    # ── Step 1: detect corners in each saved image ──────────────────────────
    obj_pts_list = []
    img_pts_list = []
    valid_samples = []

    for s in samples:
        img_path = DATA_DIR / s["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot load {img_path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect Charuco
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is None or len(charuco_corners) < 6:
            # Fallback to the detection we already stored in the JSON
            if "charuco_corners" in s and s["charuco_corners"] and len(s["charuco_corners"]) >= 6:
                charuco_corners = np.array(s["charuco_corners"], dtype=np.float32).reshape(-1, 1, 2)
                charuco_ids = np.array(s["charuco_ids"], dtype=np.int32).reshape(-1, 1)
            else:
                print(f"  [SKIP] Board not found or <6 corners in {img_path.name}")
                continue

        # Get matching 3D object points from the board definition
        obj_pts = charuco_brD.getChessboardCorners()[charuco_ids.flatten()]

        obj_pts_list.append(np.array(obj_pts, dtype=np.float32))
        img_pts_list.append(np.array(charuco_corners, dtype=np.float32))
        valid_samples.append(s)

    n = len(valid_samples)
    print(f"\n  Valid samples (board found): {n}/{len(samples)}")
    if n < 5:
        print("  ERROR: Need at least 5 valid samples. Collect more data.")
        return

    # ── Step 2: solve board → camera pose for each sample ────────────────────
    h, w = cv2.imread(str(DATA_DIR / valid_samples[0]["image"])).shape[:2]
    
    # Sanitize inputs: if distortion is crazy (~500), zero it out
    if np.any(np.abs(dist_coeffs) > 10.0):
        print("  [WARN] Input distortion coefficients look extreme (>10.0). Zeroing out for calibration.")
        dist_coeffs = np.zeros_like(dist_coeffs)

    _, camera_matrix_opt, dist_coeffs_opt, rvecs_b2c, tvecs_b2c = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, (w, h),
        camera_matrix.copy(), dist_coeffs.copy(),
        flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # Use optimized parameters for error calculation and final saving
    camera_matrix = camera_matrix_opt
    dist_coeffs = dist_coeffs_opt

    # Per-sample reprojection errors
    per_err = reprojection_errors(
        obj_pts_list, img_pts_list, rvecs_b2c, tvecs_b2c,
        camera_matrix, dist_coeffs)

    print(f"\n  Per-sample board reprojection errors (px):")
    bad_ids = []
    for i, (s, e) in enumerate(zip(valid_samples, per_err)):
        flag = "  <- WARN >3px" if e > 3.0 else ""
        print(f"    #{s['id']:03d}  {e:.2f} px{flag}")
        if e > 3.0:
            bad_ids.append(i)
    mean_err = float(np.mean(per_err))
    print(f"  Mean: {mean_err:.2f} px")
    if bad_ids:
        print(f"\n  WARNING: {len(bad_ids)} sample(s) have error > 3 px - consider removing them.")
        print(f"  IDs: {[valid_samples[i]['id'] for i in bad_ids]}")

    # ── Step 3: build gripper-to-base lists ──────────────────────────────────
    R_g2b_list = []
    t_g2b_list = []
    for s in valid_samples:
        x, y, z, rx, ry, rz = s["tcp_pose"]
        R = rotvec_to_matrix(rx, ry, rz)
        R_g2b_list.append(R)
        t_g2b_list.append(np.array([[x], [y], [z]], dtype=np.float64))

    # ── Step 4: run all hand-eye methods ────────────────────────────────────
    print(f"\n  Running hand-eye methods:")
    print(f"  {'Method':<14}  {'t_norm (m)':<12}  {'rot det':<10}  {'Status'}")
    print(f"  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}")

    results = {}
    for name, flag in METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(
                R_g2b_list, t_g2b_list,
                rvecs_b2c, tvecs_b2c,
                method=flag)
            det   = abs(np.linalg.det(R_c2g))
            tnorm = np.linalg.norm(t_c2g) * 1000  # mm
            ok    = "OK" if 0.999 < det < 1.001 else "BAD ROT"
            print(f"  {name:<14}  {tnorm:>8.1f} mm   {det:.5f}   {ok}")
            if ok == "OK":
                results[name] = (R_c2g, t_c2g)
        except Exception as e:
            print(f"  {name:<14}  FAILED: {e}")

    if not results:
        print("\n  ERROR: All methods failed. Collect more diverse samples.")
        return

    # ── Step 5: pick method (Median Selection for Stability) ────────────
    print(f"\n  Evaluating methods by board consistency (lower is better):")
    scored_results = []
    for name, (R, t) in results.items():
        consistency_err, mean_pos = eval_handeye(R, t, R_g2b_list, t_g2b_list, rvecs_b2c, tvecs_b2c)
        print(f"    {name:<12}: {consistency_err*1000:6.2f} mm | Board at ({mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f})")
        scored_results.append((consistency_err, name, R, t))
    
    # Use Median of translations as the "safe" anchor (user reports it was closer)
    t_stack = np.stack([v[1].ravel() for v in results.values()])
    t_median = np.median(t_stack, axis=0)
    best_name = min(results, key=lambda n: np.linalg.norm(results[n][1].ravel() - t_median))
    
    R_best, t_best = results[best_name]
    print(f"\n  Best method (Median anchor): {best_name}")

    # ── Step 6: print and save ────────────────────────────────────────────────
    print(f"\n  R_cam2tcp:\n{np.round(R_best, 5)}")
    print(f"\n  t_cam2tcp (mm):  dx={t_best[0,0]*1000:+.2f}  "
          f"dy={t_best[1,0]*1000:+.2f}  dz={t_best[2,0]*1000:+.2f}")

    # Sanity check for ~40 deg downward camera
    pitch_deg = np.degrees(np.arcsin(-R_best[2, 0]))
    print(f"\n  Camera pitch estimate: ~{pitch_deg:.1f} deg  (expect ~-40 deg for downward mount)")

    # Sanity check for Z direction (Eye-in-Hand should be positive Z)
    if t_best[2, 0] < 0:
        print("\n  [CRITICAL WARNING] Z-translation is NEGATIVE. The TCP is calculated to be BEHIND the camera.")
        print("  This usually means the calibration failed. Check for blurry images or wrong board size.")

    all_methods_out = {
        name: {
            "R": res[0].tolist(),
            "t": res[1].ravel().tolist()
        }
        for name, res in results.items()
    }

    output = {
        "R_cam2tcp":             R_best.tolist(),
        "t_cam2tcp":             t_best.ravel().tolist(),
        "best_method":           best_name,
        "all_methods":           all_methods_out,
        "board_reproj_mean_px":  round(mean_err, 4),
        "intrinsics": {
            "fx": camera_matrix[0, 0],
            "fy": camera_matrix[1, 1],
            "cx": camera_matrix[0, 2],
            "cy": camera_matrix[1, 2],
        },
        "distortion":    dist_coeffs.ravel().tolist(),
        "n_samples":     n,
        "sample_ids":    [s["id"] for s in valid_samples],
        "timestamp":     time.ctime(),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[SUCCESS] Saved to {OUT_FILE}")
    print("  Next: run verify_calibration.py to check the result visually.")


if __name__ == "__main__":
    main()
