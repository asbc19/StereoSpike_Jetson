#!/usr/bin/env python3
"""
Power test for StereoSpike on NVIDIA Jetson AGX Orin  (per-rail, raw logging)
Andres Brito
2026-11-19

WHAT THIS SCRIPT DOES
  - Preloads the WHOLE test set (860 binarized frames) onto the GPU ONCE as a fixed
    input pool (same frames, same order, for every repetition and every run).
  - For each run (model x precision) and each repetition:
        ACTIVE window : run the pool back-to-back, single-stream, for ACTIVE_PASSES
                        full passes -> N = ACTIVE_PASSES * 860 inferences.
        IDLE window   : GPU quiet, matched warm state, for IDLE_S seconds
                        (after a short discarded SETTLE_S).
  - Reads the on-module INA3221 rails via tegrastats in the background and slices
    each window by wall-clock time to get mean power per rail.
  - Logs RAW values per rail (no dynamic subtraction is done here; you derive
    dynamic = active - idle later):
        {RAIL}_P_active_W, {RAIL}_P_idle_W, {RAIL}_E_active_J, {RAIL}_E_idle_J
    plus shared columns (N, window durations, sensor sample counts, temperatures).
  - Energy is WINDOW-LEVEL:  E = P_mean * T_window  (divide by N later for per-inf).

  Rails exposed by tegrastats on this board (3): VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0.

OUTPUT (per run, in ./results/logs, sharing one timestamp stem):
    power_test_{model}_{precision}_{timestamp}.csv   (raw, one row per repetition)
    power_test_{model}_{precision}_{timestamp}.log   (config + summary)

OPERATING POINT (set OUTSIDE this script, before running):
    sudo nvpmodel -m 1 && sudo jetson_clocks        # MODE_15W, clocks locked

LATENCY (added 2026-11-19)
  - Each inference in the ACTIVE window is also timed: perf_counter around
    net(x) + cuda.synchronize. Inputs are already on the GPU, so only compute
    is timed (comparable to the FPGA on-board timer).
  - Per rep: lat_mean/std/min_ms and T_active_over_N_ms. Summary: pooled mean
    over all N_REPS x N inferences (value used in the paper), median of the
    rep means, and T/N.
  - Check: lat_mean must be at or just below T/N (~1% gap = Python loop,
    reset_net and autocast overhead between timed calls).
  - Assumptions: single stream, batch 1, synchronize after each inference
    (no CPU/GPU overlap), so latency = 1/throughput minus loop overhead.
    tegrastats runs in the background, so its small CPU load is included.
  - Latency and energy come from the same window -> consistent EDP.
    Report the mean (not the median), since energy averages the whole window.

RUN:
    python power_test_jetson.py
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import os
import re
import copy
import time
import csv
import shutil
import platform
import threading
import subprocess
import contextlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, surrogate
from network.SNN_models_simpl import Simplified_StereoSpike_NoSkip, NeuromorphicNet

# ----------------------------------------------------------------------------
# FP16 compatibility shim for torch.heaviside (same as in the latency script).
# The model calls torch.heaviside(conv_out - 1., <fp32 scalar>); under autocast the
# conv output is FP16 and heaviside requires matching dtypes. This coerces the scalar
# to the operand dtype. NO-OP in FP32; leaves the shared model file untouched.
# ----------------------------------------------------------------------------
_orig_heaviside = torch.heaviside
def _heaviside_dtype_safe(input, values, *args, **kwargs):
    if torch.is_tensor(values) and values.dtype != input.dtype:
        values = values.to(input.dtype)
    return _orig_heaviside(input, values, *args, **kwargs)
torch.heaviside = _heaviside_dtype_safe


# ============================================================================
#                    >>>  VARIABLES YOU WILL CHANGE LATER  <<<
# ============================================================================
# Repetitions per run. 1 now (pipeline test); set to 10 for the real measurement.
N_REPS = 10

# Active window length, expressed as full passes over the 860-frame pool.
#   N (inferences) = ACTIVE_PASSES * pool_size.  1 pass ~= 2-5 min depending on model.
ACTIVE_PASSES = 1

# Idle window (seconds) and the discarded settle before it. 30 s chosen so the
# slow-refreshing SYS_5V0 rail (~4 s) still gets several updates.
IDLE_S   = 30.0
SETTLE_S = 5.0

# Untimed warm-up inferences before the reps (warms CUDA/cuDNN kernels per run).
WARMUP_SAMPLES = 20

# Which runs to execute. Comment out lines to run fewer.
# For now: one repetition of EACH of the four (full/pruned x fp32/fp16).
RUN_MATRIX = [
    ("full",   False, "fp32"),
    ("full",   False, "fp16"),
    ("pruned", True,  "fp32"),
    ("pruned", True,  "fp16"),
]
# ============================================================================


# ----------------------------------------------------------------------------
# Other fixed configuration
# ----------------------------------------------------------------------------
PRETRAINED_CKPT_PATH = './results/checkpoints/simplified_stereospike_NoSkip.pth'
TEST_SET_PATH        = './datasets/MVSEC/data/indoor_post/test_set.pt'
LOG_DIR              = './results/logs/power'

TEGRASTATS_INTERVAL_MS = 100    # tegrastats polling period (fine; the mean is time-weighted)
POOL_SIZE              = 860     # frames to preload (full test set)

# Rails as they appear in tegrastats, and the short tokens used as CSV column PREFIXES.
RAIL_TOKENS = ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]
RAIL_SHORT  = {"VDD_GPU_SOC": "GPU_SOC", "VDD_CPU_CV": "CPU_CV", "VIN_SYS_5V0": "SYS_5V0"}
# Rails reported in the summary (median +/- spread): GPU_SOC and SYS_5V0 only.
SUMMARY_RAILS = ["VDD_GPU_SOC", "VIN_SYS_5V0"]


# ----------------------------------------------------------------------------
# Pruned model (identical construction to the original / latency script)
# ----------------------------------------------------------------------------
class Pruned_Simplified_StereoSpike_NoSkip(NeuromorphicNet):
    """Keeps only encoder + bottleneck + deconv4 + predict_depth4 (single depth output)."""
    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.bottom = copy.deepcopy(full_model.bottom)
        self.conv1  = copy.deepcopy(full_model.conv1)
        self.conv2  = copy.deepcopy(full_model.conv2)
        self.conv3  = copy.deepcopy(full_model.conv3)
        self.conv4  = copy.deepcopy(full_model.conv4)
        self.bottleneck = copy.deepcopy(full_model.bottleneck)
        self.deconv4 = copy.deepcopy(full_model.deconv4)
        self.predict_depth4 = copy.deepcopy(full_model.predict_depth4)

    def forward(self, x):
        frame = x[:, 0, :, :, :]
        out_bottom = torch.heaviside(self.bottom(frame) - 1., torch.tensor(0.0, device=x.device))
        out_conv1  = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        out_conv2  = torch.heaviside(self.conv2(out_conv1) - 1., torch.tensor(0.0, device=x.device))
        out_conv3  = torch.heaviside(self.conv3(out_conv2) - 1., torch.tensor(0.0, device=x.device))
        out_conv4  = torch.heaviside(self.conv4(out_conv3) - 1., torch.tensor(0.0, device=x.device))
        out_bottleneck = self.bottleneck(out_conv4)
        out_deconv4 = torch.heaviside(self.deconv4(out_bottleneck) - 1., torch.tensor(0.0, device=x.device))
        depth4 = self.predict_depth4(out_deconv4)
        return depth4, [out_bottleneck, out_deconv4]

Pruned_Simplified_StereoSpike_NoSkip.reformat_input_data = staticmethod(
    Simplified_StereoSpike_NoSkip.reformat_input_data
)


def load_pretrained_subset(pretrained_sd, model, verbose=False):
    """Copy tensors from a full-model state_dict into the pruned model when key+shape match."""
    model_sd = model.state_dict()
    copied, skipped = [], []
    normalized = {}
    for k, v in pretrained_sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        normalized[nk] = v
    for key, tensor in model_sd.items():
        if key in normalized and normalized[key].shape == tensor.shape:
            tensor.copy_(normalized[key]); copied.append(key)
        else:
            skipped.append(key)
    if verbose:
        print(f"  Pruned model: loaded {len(copied)} tensors, skipped {len(skipped)}.")


# ----------------------------------------------------------------------------
# Operating-point / environment capture (READ-ONLY, no sudo; sysfs based)
# ----------------------------------------------------------------------------
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def _mhz(path):
    raw = _read(path)
    try:
        return round(int(raw) / 1e6, 1)
    except Exception:
        return raw

def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def read_gpu_freq_mhz():
    return _mhz('/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq')

def capture_session_manifest(device):
    """Snapshot operating point + software stack ONCE; written into every run's log."""
    m = {}
    m['platform']    = platform.platform()
    m['python']      = platform.python_version()
    m['torch']       = torch.__version__
    m['torch_cuda']  = torch.version.cuda
    m['cudnn']       = torch.backends.cudnn.version()
    try:
        import spikingjelly
        m['spikingjelly'] = getattr(spikingjelly, '__version__', 'unknown')
    except Exception as e:
        m['spikingjelly'] = f"<{e}>"
    m['device']      = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    m['l4t_release'] = _read('/etc/nv_tegra_release')
    m['jetpack']     = _run(['bash', '-c',
                             "apt-cache show nvidia-jetpack 2>/dev/null | grep -m1 Version"])
    # operating point (read from the running system; no root)
    m['nvpmodel_status_file'] = _read('/var/lib/nvpmodel/status')
    m['nvpmodel_q']  = _run(['nvpmodel', '-q'])   # only warns without root; mode still prints
    m['cpu_online']  = _read('/sys/devices/system/cpu/online')
    m['gpu_min_MHz'] = _mhz('/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq')
    m['gpu_max_MHz'] = _mhz('/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq')
    m['cpu_cur_kHz'] = {c: _read(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
                        for c in range(4)}
    m['tpc_pg_mask'] = _read('/sys/devices/platform/gpu.0/tpc_pg_mask')
    m['cudnn_benchmark']      = torch.backends.cudnn.benchmark
    m['tegrastats_interval_ms'] = TEGRASTATS_INTERVAL_MS
    return m


# ----------------------------------------------------------------------------
# tegrastats background sampler: parses per-rail power (mW) + junction temp (tj).
# Runs for the whole session; each window is sliced out later by wall-clock time.
# ----------------------------------------------------------------------------
_POWER_RE = re.compile(r'([A-Za-z0-9_]+)\s+(\d+)mW/\d+mW')   # token + FIRST (instantaneous) mW
_TJ_RE    = re.compile(r'tj@([\d.]+)C')

class TegrastatsSampler(threading.Thread):
    def __init__(self, interval_ms):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples = []            # list of (wall_ts, {rail: mW}, tj_C)
        self._stop = threading.Event()
        self._proc = None

    def run(self):
        cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            powers = {tok: int(mw) for tok, mw in _POWER_RE.findall(line)}
            rails = {r: powers[r] for r in RAIL_TOKENS if r in powers}
            if not rails:
                continue
            tj_m = _TJ_RE.search(line)
            tj = float(tj_m.group(1)) if tj_m else None
            self.samples.append((time.time(), rails, tj))

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try: self._proc.terminate()
            except Exception: pass


def window_stats(samples, t_start, t_end):
    """Mean power per rail (W) + mean tj (C) + sample count over [t_start, t_end]."""
    sel = [(rails, tj) for (ts, rails, tj) in samples if t_start <= ts <= t_end]
    if not sel:
        return None, None, 0
    rail_w = {}
    for tok in RAIL_TOKENS:
        vals = [rails[tok] for (rails, _) in sel if tok in rails]
        rail_w[tok] = (sum(vals) / len(vals)) / 1000.0 if vals else None   # mW -> W
    tjs = [tj for (_, tj) in sel if tj is not None]
    tj_mean = (sum(tjs) / len(tjs)) if tjs else None
    return rail_w, tj_mean, len(sel)


# ----------------------------------------------------------------------------
# Input pool: preload the whole test set onto the GPU once (no per-iter H2D).
# ----------------------------------------------------------------------------
def build_input_pool(device, n_pool):
    test_set = torch.load(TEST_SET_PATH, weights_only=False)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                         drop_last=True, pin_memory=True)
    reformat = Simplified_StereoSpike_NoSkip.reformat_input_data
    pool = []
    with torch.no_grad():
        for sample in loader:
            _, wl, wr, tl, tr, _ = sample
            wl = wl.to(device, dtype=torch.float); wr = wr.to(device, dtype=torch.float)
            tl = tl.to(device, dtype=torch.float); tr = tr.to(device, dtype=torch.float)
            _, test_chunks = reformat(wl, wr, tl, tr)
            pool.append((test_chunks >= 1).to(test_chunks.dtype))   # binarized, stays on GPU
            if len(pool) >= n_pool:
                break
    return pool


# ----------------------------------------------------------------------------
# Window runners
# ----------------------------------------------------------------------------
def _autocast_ctx(use_fp16):
    return (torch.autocast(device_type='cuda', dtype=torch.float16)
            if use_fp16 else contextlib.nullcontext())

def run_active_window(net, pool, passes, use_fp16, device):
    """Run the pool back-to-back, single-stream, for `passes` full passes.
    Returns (N, t0, t1, lat_ms), lat_ms = per-inference time of net(x) + synchronize."""
    torch.cuda.synchronize(device)
    lat_ms = []
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for _ in range(passes):
            for x in pool:
                functional.reset_net(net)          # no-op for this feedforward CNN; kept for parity
                s0 = time.perf_counter()           # GPU already idle (synchronized below)
                with _autocast_ctx(use_fp16):
                    _ = net(x)
                torch.cuda.synchronize(device)     # single-stream: finish before the next
                lat_ms.append((time.perf_counter() - s0) * 1e3)
                n += 1
    t1 = time.time()
    return n, t0, t1, lat_ms

def measure_idle(idle_s, settle_s, device):
    """GPU quiet, warm, matched state. Returns (t0, t1) of the measured idle window."""
    torch.cuda.synchronize(device)
    time.sleep(settle_s)                # discarded: let the active tail drain from the sensors
    t0 = time.time()
    time.sleep(idle_s)
    t1 = time.time()
    return t0, t1


# ----------------------------------------------------------------------------
# Log writer
# ----------------------------------------------------------------------------
def write_log(log_path, manifest, run_info, summary):
    def section(t): return f"\n===== {t} =====\n"
    with open(log_path, 'w') as f:
        f.write("StereoSpike power test - run log\n")
        f.write(f"written: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(section("RUN"))
        for k, v in run_info.items(): f.write(f"{k}: {v}\n")
        f.write(section("OPERATING POINT + STACK (read at session start)"))
        for k, v in manifest.items(): f.write(f"{k}: {v}\n")
        f.write(section("SUMMARY (median +/- std over reps; GPU_SOC and SYS_5V0 only)"))
        for k, v in summary.items(): f.write(f"{k}: {v}\n")


# ----------------------------------------------------------------------------
# CSV schema (rail token as PREFIX; window-level energy only)
# ----------------------------------------------------------------------------
SHARED_COLS = ["model", "precision", "rep", "passes", "N",
               "T_active_s", "T_idle_s", "n_samp_active", "n_samp_idle",
               "tj_active_C", "tj_idle_C",
               "lat_mean_ms", "lat_std_ms", "lat_min_ms", "T_active_over_N_ms"]

def rail_cols():
    cols = []
    for tok in RAIL_TOKENS:
        s = RAIL_SHORT[tok]
        cols += [f"{s}_P_active_W", f"{s}_P_idle_W", f"{s}_E_active_J", f"{s}_E_idle_J"]
    return cols

CSV_HEADER = SHARED_COLS + rail_cols()


# ----------------------------------------------------------------------------
# One power run (model x precision) -> one CSV + one LOG
# ----------------------------------------------------------------------------
def run_power_test(net, model_name, is_pruned, precision, pool, device,
                   sampler, manifest, timestamp):
    use_fp16 = (precision == "fp16")
    stem = f"power_test_{model_name}_{precision}_{timestamp}"
    csv_path = os.path.join(LOG_DIR, stem + ".csv")
    log_path = os.path.join(LOG_DIR, stem + ".log")
    print(f"\n>>> Power run: {model_name} / {precision}  ->  {stem}")
    net.eval()

    # --- warm-up (untimed): warm kernels for THIS model/precision ---
    with torch.no_grad():
        for i, x in enumerate(pool):
            functional.reset_net(net)
            with _autocast_ctx(use_fp16):
                _ = net(x)
            if i + 1 >= WARMUP_SAMPLES:
                break
    torch.cuda.synchronize(device)

    start_gpu_freq = read_gpu_freq_mhz()

    # per-rep values for the two summary rails (for median +/- spread)
    summ = {tok: {"P_active": [], "P_idle": [], "E_active": [], "E_idle": []}
            for tok in SUMMARY_RAILS}
    lat_all, lat_repmean, tn_rep = [], [], []     # latency from the same active windows

    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_HEADER)
        writer.writeheader()

        for rep in range(1, N_REPS + 1):
            # ---- ACTIVE window ----
            N, a0, a1, lat = run_active_window(net, pool, ACTIVE_PASSES, use_fp16, device)
            lat = np.asarray(lat)
            T_active = a1 - a0
            # ---- IDLE window (warm, matched) ----
            i0, i1 = measure_idle(IDLE_S, SETTLE_S, device)
            T_idle = i1 - i0

            # ---- slice the sampler stream for each window ----
            act_w, act_tj, act_n = window_stats(sampler.samples, a0, a1)
            idl_w, idl_tj, idl_n = window_stats(sampler.samples, i0, i1)
            if act_w is None or idl_w is None:
                print(f"  rep {rep}: no power samples in a window -> check tegrastats. Skipped.")
                continue

            # ---- assemble the raw row (window-level energy = P_mean * T_window) ----
            row = {
                "model": model_name, "precision": precision, "rep": rep,
                "passes": ACTIVE_PASSES, "N": N,
                "T_active_s": round(T_active, 4), "T_idle_s": round(T_idle, 4),
                "n_samp_active": act_n, "n_samp_idle": idl_n,
                "tj_active_C": round(act_tj, 2) if act_tj is not None else "",
                "tj_idle_C":   round(idl_tj, 2) if idl_tj is not None else "",
                "lat_mean_ms": round(float(lat.mean()), 4),
                "lat_std_ms":  round(float(lat.std(ddof=1)), 4),
                "lat_min_ms":  round(float(lat.min()), 4),
                "T_active_over_N_ms": round(T_active / N * 1e3, 4),
            }
            lat_all.extend(lat.tolist()); lat_repmean.append(float(lat.mean()))
            tn_rep.append(T_active / N * 1e3)
            for tok in RAIL_TOKENS:
                s = RAIL_SHORT[tok]
                pa = act_w.get(tok); pi = idl_w.get(tok)
                ea = pa * T_active if pa is not None else None
                ei = pi * T_idle   if pi is not None else None
                row[f"{s}_P_active_W"] = round(pa, 4) if pa is not None else ""
                row[f"{s}_P_idle_W"]   = round(pi, 4) if pi is not None else ""
                row[f"{s}_E_active_J"] = round(ea, 4) if ea is not None else ""
                row[f"{s}_E_idle_J"]   = round(ei, 4) if ei is not None else ""
                if tok in SUMMARY_RAILS and pa is not None and pi is not None:
                    summ[tok]["P_active"].append(pa); summ[tok]["P_idle"].append(pi)
                    summ[tok]["E_active"].append(ea); summ[tok]["E_idle"].append(ei)

            writer.writerow(row); fcsv.flush()   # crash-safe raw data
            print(f"  rep {rep}: N={N}  GPU_SOC P_act={act_w.get('VDD_GPU_SOC'):.3f}W "
                  f"P_idle={idl_w.get('VDD_GPU_SOC'):.3f}W  "
                  f"T_act={T_active:.1f}s  tj_act={row['tj_active_C']}C  "
                  f"(samp act/idle={act_n}/{idl_n})  "
                  f"lat={lat.mean():.3f}+/-{lat.std(ddof=1):.3f}ms vs T/N={T_active / N * 1e3:.3f}ms")

    end_gpu_freq = read_gpu_freq_mhz()

    # ---- summary: median +/- std over reps, GPU_SOC and SYS_5V0 only ----
    def med_std(v):
        a = np.asarray(v, dtype=float)
        if a.size == 0: return (None, None)
        return (round(float(np.median(a)), 4),
                round(float(np.std(a, ddof=1)), 4) if a.size > 1 else 0.0)

    summary = {}
    for tok in SUMMARY_RAILS:
        s = RAIL_SHORT[tok]
        for key in ["P_active", "P_idle", "E_active", "E_idle"]:
            med, sd = med_std(summ[tok][key])
            unit = "W" if key.startswith("P") else "J"
            summary[f"{s}_{key}_{unit}_median"] = med
            summary[f"{s}_{key}_{unit}_std"]    = sd
    la = np.asarray(lat_all)
    summary["latency_ms_mean_pooled"] = round(float(la.mean()), 4) if la.size else None
    summary["latency_ms_std_pooled"]  = round(float(la.std(ddof=1)), 4) if la.size > 1 else None
    summary["latency_ms_repmean_median"], summary["latency_ms_repmean_std"] = med_std(lat_repmean)
    summary["T_active_over_N_ms_median"], summary["T_active_over_N_ms_std"] = med_std(tn_rep)

    run_info = {
        "model": model_name, "pruned": is_pruned, "precision": precision,
        "fp16_method": "torch.autocast (fp16 matmul / fp32 accumulate)" if use_fp16 else "native fp32",
        "n_reps": N_REPS, "active_passes": ACTIVE_PASSES, "pool_size": len(pool),
        "N_per_rep": ACTIVE_PASSES * len(pool),
        "idle_s": IDLE_S, "settle_s": SETTLE_S, "warmup_samples": WARMUP_SAMPLES,
        "csv_file": os.path.basename(csv_path),
        "note": "dynamic = active - idle is derived at analysis time (not stored here)",
        "latency_method": "perf_counter around net(x) + cuda.synchronize, inside the active window",
        "gpu_freq_MHz_start": start_gpu_freq, "gpu_freq_MHz_end": end_gpu_freq,
    }
    write_log(log_path, manifest, run_info, summary)
    if start_gpu_freq != end_gpu_freq:
        print("    [!] GPU frequency changed during the run -> possible throttling, check the log.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: this test must run on the Jetson GPU.")
    if shutil.which("tegrastats") is None:
        raise SystemExit("tegrastats not found: cannot measure power.")

    torch.backends.cudnn.benchmark = False    # deterministic kernels (consistent with latency test)
    os.makedirs(LOG_DIR, exist_ok=True)
    device = torch.device('cuda:0')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Session timestamp: {timestamp}")

    manifest = capture_session_manifest(device)

    # --- build models (FP32 weights; FP16 handled by autocast at runtime) ---
    full = Simplified_StereoSpike_NoSkip(
        input_chans=2 * 2 * 1, tau=3., v_threshold=1.0, v_reset=0.0,
        use_plif=True, multiply_factor=10.,
        surrogate_function=surrogate.ATan(), learnable_biases=False
    ).to(device)
    full.load_state_dict(torch.load(PRETRAINED_CKPT_PATH, map_location=device, weights_only=False))
    full.eval()

    pruned = Pruned_Simplified_StereoSpike_NoSkip(full).to(device)
    ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location=device, weights_only=False)
    load_pretrained_subset(ckpt.get("state_dict", ckpt), pruned, verbose=True)
    pruned.eval()
    nets = {"full": full, "pruned": pruned}

    # --- preload the fixed input pool ONCE (shared by every rep and run) ---
    print(f"Preloading pool of {POOL_SIZE} frames ...")
    pool = build_input_pool(device, POOL_SIZE)
    print(f"Pool size: {len(pool)} frames")

    # --- start the continuous power sampler ---
    sampler = TegrastatsSampler(TEGRASTATS_INTERVAL_MS)
    sampler.start()
    time.sleep(2.0)                                  # let tegrastats spin up
    if not sampler.samples:
        sampler.stop()
        raise SystemExit("No tegrastats samples. Try running with sudo, or check rail names.")

    # --- run the matrix ---
    try:
        for model_name, is_pruned, precision in RUN_MATRIX:
            run_power_test(nets[model_name], model_name, is_pruned, precision,
                           pool, device, sampler, manifest, timestamp)
    finally:
        sampler.stop()

    print(f"\nDone. Files written to: {LOG_DIR}")


if __name__ == "__main__":
    main()