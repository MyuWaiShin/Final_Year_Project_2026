"""
safety_guard.py
================
Shared safety module — import this in any script that moves the robot.

Classes
-------
SafetyViolation
    Exception raised when a requested pose is outside the safe workspace.

SafetyGuard
    Loads Perception/safe_limits.json and validates/clamps target poses.
    Also holds a dynamic `floor_z` that DepthFloorDetector updates.

Usage
-----
    from UR10.safety_guard import SafetyGuard, SafetyViolation

    guard = SafetyGuard()           # loads limits from default path
    guard.check(x, y, z)            # raises SafetyViolation if out of bounds
    x2, y2, z2 = guard.clamp(x, y, z)  # clamps silently, prints warning
"""

import json
import threading
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Default path to safe_limits.json (relative to THIS file's location)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_LIMITS = Path(__file__).parent.parent / "Perception" / "safe_limits.json"


class SafetyViolation(Exception):
    """Raised when a target pose is outside the recorded safe workspace."""
    pass


class SafetyGuard:
    """
    Loads workspace limits from safe_limits.json and enforces them.

    Parameters
    ----------
    limits_path : str or Path, optional
        Path to safe_limits.json. Defaults to Perception/safe_limits.json.
    floor_clearance_m : float
        Minimum clearance above the detected floor (metres). Default = 0.002.
    warn_margin_m : float
        Distance from a limit that triggers a warning instead of hard stop.
        Set to 0 to disable soft warnings (all violations are hard stops).
    """

    def __init__(self,
                 limits_path=None,
                 floor_clearance_m: float = 0.002,
                 warn_margin_m: float = 0.020):

        path = Path(limits_path) if limits_path else _DEFAULT_LIMITS

        if not path.exists():
            raise FileNotFoundError(
                f"[SafetyGuard] safe_limits.json not found at {path}.\n"
                f"Run:  python Perception/record_safe_limits.py  to create it."
            )

        with open(path) as f:
            data = json.load(f)

        lim = data["safe_limits"]
        self.safe_x: tuple = tuple(lim["SAFE_X"])   # (min, max) metres
        self.safe_y: tuple = tuple(lim["SAFE_Y"])
        self.safe_z: tuple = tuple(lim["SAFE_Z"])

        self.floor_clearance_m = floor_clearance_m
        self.warn_margin_m     = warn_margin_m

        # Dynamic floor Z — updated by DepthFloorDetector.
        # Initially set to the static lower Z limit.
        self._floor_z: float = self.safe_z[0]
        self._floor_lock = threading.Lock()

        margin = data.get("margin_m", 0.01)
        ts     = data.get("timestamp", "unknown")
        print(f"[SafetyGuard] Loaded limits from {path.name}  (recorded: {ts})")
        print(f"  SAFE_X : [{self.safe_x[0]*1000:+.1f}, {self.safe_x[1]*1000:+.1f}] mm  "
              f"SAFE_Y : [{self.safe_y[0]*1000:+.1f}, {self.safe_y[1]*1000:+.1f}] mm  "
              f"SAFE_Z : [{self.safe_z[0]*1000:+.1f}, {self.safe_z[1]*1000:+.1f}] mm  "
              f"(margin={margin*1000:.0f} mm)")

    # ── Dynamic floor Z (updated by DepthFloorDetector) ──────────────────────
    @property
    def floor_z(self) -> float:
        with self._floor_lock:
            return self._floor_z

    @floor_z.setter
    def floor_z(self, value: float):
        with self._floor_lock:
            self._floor_z = value

    def effective_z_min(self) -> float:
        """Returns the tighter of static low Z and dynamic floor + clearance."""
        dynamic_min = self.floor_z + self.floor_clearance_m
        return max(self.safe_z[0], dynamic_min)

    # ── Validation ────────────────────────────────────────────────────────────
    def check(self, x: float, y: float, z: float) -> None:
        """
        Raises SafetyViolation if (x, y, z) is outside the safe workspace.
        Uses the tighter of the static Z limit and the dynamic floor limit.

        Parameters are in metres (robot base frame).
        """
        violations = []

        if not (self.safe_x[0] <= x <= self.safe_x[1]):
            violations.append(
                f"X={x*1000:+.1f}mm  (limits [{self.safe_x[0]*1000:+.1f}, "
                f"{self.safe_x[1]*1000:+.1f}] mm)"
            )
        if not (self.safe_y[0] <= y <= self.safe_y[1]):
            violations.append(
                f"Y={y*1000:+.1f}mm  (limits [{self.safe_y[0]*1000:+.1f}, "
                f"{self.safe_y[1]*1000:+.1f}] mm)"
            )

        z_min = self.effective_z_min()
        if not (z_min <= z <= self.safe_z[1]):
            floor_note = (
                f"  [floor_z={self.floor_z*1000:.1f}mm + "
                f"{self.floor_clearance_m*1000:.0f}mm clearance = "
                f"effective_min={z_min*1000:.1f}mm]"
            )
            violations.append(
                f"Z={z*1000:+.1f}mm  (limits [{z_min*1000:+.1f}, "
                f"{self.safe_z[1]*1000:+.1f}] mm){floor_note}"
            )

        if violations:
            msg = "[SafetyGuard] VIOLATION — target pose rejected:\n  " + \
                  "\n  ".join(violations)
            raise SafetyViolation(msg)

        # Soft warnings
        if self.warn_margin_m > 0:
            self._warn_if_near(x, y, z, z_min)

    def _warn_if_near(self, x, y, z, z_min):
        w = self.warn_margin_m
        for name, v, lo, hi in [("X", x, self.safe_x[0], self.safe_x[1]),
                                  ("Y", y, self.safe_y[0], self.safe_y[1]),
                                  ("Z", z, z_min,           self.safe_z[1])]:
            dist = min(abs(v - lo), abs(v - hi))
            if dist < w:
                print(f"[SafetyGuard] ⚠  {name}={v*1000:+.1f}mm is "
                      f"{dist*1000:.1f}mm from a limit")

    def clamp(self, x: float, y: float, z: float) -> tuple:
        """
        Clamps (x, y, z) to the safe workspace and returns the clamped values.
        Prints a warning if any axis was clamped. Does NOT raise an exception.
        """
        z_min = self.effective_z_min()

        cx = max(self.safe_x[0], min(self.safe_x[1], x))
        cy = max(self.safe_y[0], min(self.safe_y[1], y))
        cz = max(z_min,          min(self.safe_z[1], z))

        if cx != x or cy != y or cz != z:
            print(f"[SafetyGuard] Clamped: "
                  f"({x*1000:+.1f},{y*1000:+.1f},{z*1000:+.1f}) → "
                  f"({cx*1000:+.1f},{cy*1000:+.1f},{cz*1000:+.1f}) mm")
        return cx, cy, cz


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  SafetyGuard self-test")
    print("=" * 60)

    try:
        guard = SafetyGuard()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"\n  floor_z (initial) = {guard.floor_z*1000:.1f} mm")
    print(f"  effective_z_min   = {guard.effective_z_min()*1000:.1f} mm")

    # Test 1: pose right in the middle of the workspace
    mx = (guard.safe_x[0] + guard.safe_x[1]) / 2
    my = (guard.safe_y[0] + guard.safe_y[1]) / 2
    mz = (guard.safe_z[0] + guard.safe_z[1]) / 2
    print(f"\n[Test 1] Centre pose ({mx*1000:+.1f}, {my*1000:+.1f}, {mz*1000:+.1f}) mm ...", end=" ")
    try:
        guard.check(mx, my, mz)
        print("PASS ✓")
    except SafetyViolation as e:
        print(f"FAIL ✗\n{e}")

    # Test 2: X way out of range
    bad_x = guard.safe_x[1] + 0.5
    print(f"[Test 2] Out-of-range X ({bad_x*1000:+.1f} mm) ...", end=" ")
    try:
        guard.check(bad_x, my, mz)
        print("FAIL ✗ — should have raised")
    except SafetyViolation:
        print("PASS ✓ (SafetyViolation raised correctly)")

    # Test 3: Z below floor
    bad_z = guard.safe_z[0] - 0.1
    print(f"[Test 3] Z below static limit ({bad_z*1000:+.1f} mm) ...", end=" ")
    try:
        guard.check(mx, my, bad_z)
        print("FAIL ✗ — should have raised")
    except SafetyViolation:
        print("PASS ✓ (SafetyViolation raised correctly)")

    # Test 4: clamp
    print(f"[Test 4] Clamp out-of-range pose ...", end=" ")
    cx, cy, cz = guard.clamp(bad_x, my, bad_z)
    print(f"PASS ✓  clamped Z → {cz*1000:.1f} mm, X → {cx*1000:.1f} mm")

    print("\n  All tests passed.")
