#!/usr/bin/env python3
"""
Probe the Jetson AGX Orin on-module INA3221 power-rail update behavior, to size
the power-test windows (especially the short IDLE window) from real data.

It does NOT run the network. It launches `tegrastats` at a fine interval for a
few seconds while the board is idle, parses the per-rail power tokens, and reports
for each rail:
  - how many samples arrived and the effective sample period,
  - how many samples were DISTINCT (value actually changed) vs repeated,
    i.e. the real refresh rate behind tegrastats' polling,
  - basic idle-power stats (mean / min / max / std).

Run in the SAME operating point the idle window will use:
    sudo nvpmodel -m 1 && sudo jetson_clocks     # MODE_15W, clocks locked
    python probe_ina3221_rate.py                 # GPU otherwise idle

tegrastats does NOT require sudo for reading on most JetPack builds; if it prints
nothing, re-run with `sudo python probe_ina3221_rate.py`.
"""

import re
import time
import shutil
import subprocess
from collections import defaultdict

DURATION_S   = 8.0      # how long to sample
INTERVAL_MS  = 50       # tegrastats polling interval (ask fast; real refresh may be slower)

# AGX Orin rails we care about (tegrastats token names)
RAILS = ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0", "VDDQ_VDD2_1V8AO"]

# token like:  VDD_GPU_SOC 1234mW/1234mW   (we take the FIRST number = instantaneous)
tok_re = re.compile(r'([A-Za-z0-9_]+)\s+(\d+)mW/\d+mW')

if shutil.which("tegrastats") is None:
    raise SystemExit("tegrastats not found on PATH.")

cmd = ["tegrastats", "--interval", str(INTERVAL_MS)]
print(f"Sampling tegrastats at {INTERVAL_MS} ms for {DURATION_S:.0f} s (board should be idle)...\n")

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True, bufsize=1)

# per-rail list of (wall_time, power_mW)
series = defaultdict(list)
t0 = time.time()
try:
    for line in proc.stdout:
        now = time.time()
        if now - t0 > DURATION_S:
            break
        found = dict(tok_re.findall(line))
        for rail in RAILS:
            if rail in found:
                series[rail].append((now, int(found[rail])))
finally:
    proc.terminate()

if not series:
    raise SystemExit("No rail tokens parsed. Try `sudo python probe_ina3221_rate.py`, "
                     "or check the rail names with a raw `tegrastats` line.")

print(f"{'rail':<18}{'n':>5}{'period_ms':>11}{'distinct':>9}{'refresh_ms':>11}"
      f"{'mean_mW':>9}{'min':>7}{'max':>7}{'std':>7}")
print("-" * 84)

for rail in RAILS:
    s = series.get(rail, [])
    if len(s) < 2:
        print(f"{rail:<18}{len(s):>5}   (too few samples)")
        continue

    ts = [t for t, _ in s]
    p  = [v for _, v in s]
    span = ts[-1] - ts[0]
    period_ms = 1000.0 * span / (len(s) - 1)          # tegrastats emission period

    # count value CHANGES -> real sensor refresh behind the polling
    changes = sum(1 for i in range(1, len(p)) if p[i] != p[i - 1])
    refresh_ms = (1000.0 * span / changes) if changes > 0 else float('inf')

    mean = sum(p) / len(p)
    var  = sum((v - mean) ** 2 for v in p) / len(p)
    std  = var ** 0.5

    refresh_str = "inf" if changes == 0 else f"{refresh_ms:>7.0f}"
    print(f"{rail:<18}{len(s):>5}{period_ms:>11.1f}{changes:>9}{refresh_str:>11}"
          f"{mean:>9.0f}{min(p):>7}{max(p):>7}{std:>7.1f}")

print("\nHow to read this:")
print("  period_ms  = how often tegrastats emits a line.")
print("  distinct   = how many times a rail's value actually CHANGED in the window.")
print("  refresh_ms = real sensor refresh (span / distinct). Size the IDLE window to")
print("               contain MANY refreshes -- aim for >= ~50-100 distinct updates.")
print("  std        = idle-power noise; low std => a short idle window already gives a")
print("               stable mean.")