"""Benchmark climb_motion.lock_on: peak memory + wall time, optionally compare to a saved baseline.
usage: python tools/bench_lock_on.py CAM_PKL OUT_NPY [BASELINE_NPY]
CAM_PKL is a pickle of (base_frame, cam) from track_camera. Video path is read from BV_VIDEO."""
import ctypes, os, pickle, sys, time
from ctypes import wintypes
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import climb_motion as cm

POLYS = [  # base-frame pixels, IMG_3247 base frame (n//2)
    [(140, 230), (380, 200), (370, 310), (200, 340)],
    [(420, 210), (500, 200), (575, 320), (520, 355), (430, 260)],
    [(465, 480), (520, 455), (550, 510), (500, 560), (465, 520)],
    [(615, 980), (700, 820), (745, 860), (660, 1040)],
    [(725, 900), (800, 880), (815, 990), (770, 1020)],
    [(585, 1090), (720, 1010), (735, 1060), (640, 1130), (590, 1120)],
]

class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
        [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
         "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
         "PagefileUsage", "PeakPagefileUsage")]

def peak_mb():
    p = PMC(); p.cb = ctypes.sizeof(p)
    k = ctypes.WinDLL("kernel32", use_last_error=True); k.GetCurrentProcess.restype = wintypes.HANDLE
    ps = ctypes.WinDLL("psapi", use_last_error=True)
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.PeakWorkingSetSize / 2**20, p.PeakPagefileUsage / 2**20

if __name__ == "__main__":
    base, cam = pickle.load(open(sys.argv[1], "rb"))
    video = os.environ["BV_VIDEO"]
    t = time.time()
    out = cm.lock_on(video, cam, [np.float32(p) for p in POLYS])
    dt = time.time() - t
    ws, pf = peak_mb()
    np.save(sys.argv[2], out)
    print(f"time {dt:.1f}s  peak working set {ws:.0f} MB  peak commit {pf:.0f} MB  shape {out.shape}")
    if len(sys.argv) > 3:
        b = np.load(sys.argv[3]); d = np.abs(out - b)
        print(f"offset diff vs baseline: max {d.max():.3f} px  median {np.median(d):.4f} px  mean {d.mean():.4f}  p99 {np.percentile(d,99):.3f}")
        print("per-hold max:", d.max(axis=(0, 2)).round(3))
