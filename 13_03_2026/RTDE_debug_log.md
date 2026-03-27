# RTDE Debug Log — UR10 + RG2 Pipeline
**File:** `13_failure_detection.py`  
**Date:** 2026-03-24  
**Robot:** UR10 @ 192.168.8.102  

---

## What Failed and Why

### ❌ RTDEControlInterface — Hangs on Connect (Initial Issue)

**Symptom:**
```
[Init] Connecting RTDE control to 192.168.8.102...
# hangs forever — Ctrl+C doesn't work
```

**Root cause:**  
`RTDEControlInterface(ip)` is a C++ constructor (via Python binding).  
When it blocks, Python's signal delivery queue is frozen — Ctrl+C has no effect.  
This happens when the robot can't accept the RTDE script (e.g. EtherNet/IP enabled, robot in fault, another RTDE client connected).

**Fix applied:**  
Wrapped constructor in a daemon thread with a 15-second timeout.  
Added `signal.signal(SIGINT, lambda *_: os._exit(1))` at `__main__` so Ctrl+C always kills immediately.

---

### ❌ RTDEControlInterface — Drops Mid-Run, Reconnect Hangs (Core Issue)

**Symptom (after the ENTER prompt):**
```
RTDEReceiveInterface boost system Exception: (asio.misc:2) End of file
RTDEControlInterface: Could not receive data from robot...
RTDEControlInterface Exception: End of file
Reconnecting...
# hangs forever in RtlSleepConditionVariableCS
```

**Stack trace showed:**  
`RTDEControlInterface::reconnect → ScriptClient::connect → RtlSleepConditionVariableCS`  
The C++ reconnect logic blocked on a Windows sync primitive — unkillable from Python.

**Root cause:**  
`_play_urp()` called `self._rtde_c.stopScript()` before loading each gripper URP.  
This tells the robot to stop the RTDE keepalive script → robot drops the RTDE TCP connection.  
RTDE-control then tries to auto-reconnect internally (C++ level) → hangs.

**Why script `12_move_to_cylinder.py` works fine:**  
Script 12 connects RTDE control once, does one `moveL`, disconnects.  
It **never calls `stopScript()`** or dashboard commands while RTDE control is live.  
Script 13 was calling `stopScript()` on every single gripper command — constant conflict.

---

### ✅ Solution: Remove RTDEControlInterface Entirely

**Approach:**  
- Use only `RTDEReceiveInterface` (port 30004, read-only, no script upload, never conflicts)  
- Send all motion as raw URScript via a **persistent TCP socket to port 30002**  
- Poll `getActualTCPPose()` on RTDE receive to detect arrival (10ms poll interval)  
- Dashboard URPs (gripper) run freely — no script to suspend/restore

**Why this is NOT slower than RTDEControlInterface:**  
- `RTDEControlInterface.moveL()` also sends URScript internally  
- RTDE receive state updates at 125 Hz (8ms resolution) — same resolution we poll at  
- Removing the per-call TCP socket open/close eliminates ~5–20ms of handshake latency **per move**  
- No `stopScript()`/`reuploadScript()` round-trips = ~100ms saved per gripper cycle

---

## Delay Sources Identified and Fixed

| Source | Before | After |
|--------|--------|-------|
| `_send_urscript()` TCP socket | New socket per call (~10–20ms handshake) | Persistent socket (0ms) |
| Motion poll interval | 50ms sleep | 10ms sleep |
| `stopScript()` + `reuploadScript()` per gripper | ~100ms per URP | Removed entirely |
| RTDE control connect | Blocking forever if robot rejects | 15s timeout + clean error |

---

## Port Usage Summary

| Port | Protocol | Used For | Notes |
|------|----------|----------|-------|
| 29999 | TCP | Dashboard (URP load/play/stop) | Short-lived connections, safe |
| 30002 | TCP | Secondary interface: sensor stream + URScript send | Persistent RX socket for sensors; persistent TX socket for URScript |
| 30004 | TCP | RTDE receive (TCP pose, `isProgramRunning`) | Read-only, no script upload, stable |

---

## Key Lessons (Blog Reference)

1. **`RTDEControlInterface` uploads a URScript** to the robot on connect. Any call that stops this script (including `stopScript()` from Python, or the dashboard `stop` command) drops the TCP connection and triggers C++ reconnect.

2. **Never mix dashboard URP calls with RTDEControlInterface** unless you handle the disconnect/reconnect carefully. The Python binding's auto-reconnect is uninterruptible.

3. **Raw URScript on port 30002 is equally fast.** The robot executes it immediately. Blocking arrival detection via RTDE receive polling is the same latency as RTDEControlInterface's internal blocking (both bottleneck at RTDE's 125Hz state updates).

4. **Port 30002 supports multiple simultaneous clients.** A persistent sensor-reading socket and a persistent URScript-sending socket on port 30002 coexist without interference.

5. **Ctrl+C cannot interrupt C extension blocking calls** in Python. Use daemon threads with timeouts + `os._exit(1)` in a SIGINT handler.
