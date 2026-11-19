#!/usr/bin/env python3
"""
Andres Brito
2026-11-19

Latency + Mean Depth Error (MDE) test for StereoSpike on NVIDIA Jetson AGX Orin

WHAT THIS SCRIPT DOES
  - Runs the FULL and the PRUNED Simplified_StereoSpike_NoSkip networks over the
    WHOLE test set (batch size 1, single-stream), each in FP32 and in FP16.
    => 4 independent runs: {full, pruned} x {fp32, fp16}.
  - For every run it produces TWO files that share the same name stem:
        results/logs/latency_test_{model}_{precision}_{timestamp}.csv  (raw, per-sample)
        results/logs/latency_test_{model}_{precision}_{timestamp}.log  (config + summary)
  - Times ONLY the net(...) call (bracketed by cuda.synchronize) so latency is clean.
    MDE is computed OUTSIDE the timed region.

OPERATING POINT (must be set OUTSIDE this script, before running):
        sudo nvpmodel -m 1      # MODE_15W
        sudo jetson_clocks       # lock clocks (re-run after every boot)
  This script does NOT change the power mode; it only READS and LOGS it, and
  re-checks the GPU clock/temperature after each pass (thermal guard).

FP16 POLICY
  FP16 runs use torch.autocast (FP16 multiply / FP32 accumulate). This mirrors the
  Tensor-Core datapath and the FPGA (16-bit operands, wider accumulator), and keeps
  the SNN neuron state (membrane potentials) in FP32. We do NOT use net.half().

NOTE ON LATENCY (2026-11-19)
  - On the Jetson, PyTorch eager runs every kernel launch from the slow host
    CPU, so host activity between inferences inflates the timed net(x) call.
    The old version (tqdm bar + per-row CSV flush, 5 warm-up samples) showed
    periodic slow samples locked to the 0.1 s tqdm refresh, giving a mean
    15-25% above the back-to-back window/N.
  - Fix: tqdm disabled (USE_TQDM), CSV written once after the loop,
    20 warm-up samples. The periodic pattern disappears.
  - Remaining limitation: data loading, preprocessing and MDE run between
    timed calls and can still disturb them (e.g. sporadic slow samples
    right after warm-up).
  - Use this script for MDE only. The reported latency comes from
    power_test_jetson.py (same window as the energy).

RUN
        python latency_mde_jetson.py
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import os
import glob
import copy
import time
import csv
import platform
import subprocess
import contextlib
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# SpikingJelly: activation_based API (the one installed/working on this Jetson).
# Keep this identical on the V100 for the comparison to be valid.
from spikingjelly.activation_based import functional, surrogate

# Network + metrics (unchanged modules from the codebase)
from network.SNN_models_simpl import Simplified_StereoSpike_NoSkip, NeuromorphicNet
from network.metrics import MeanDepthError_n

# ----------------------------------------------------------------------------
# FP16 compatibility shim for torch.heaviside
# The network computes torch.heaviside(conv_out - 1., <scalar>), where <scalar> is a
# hard-coded FP32 tensor. Under autocast the conv output is FP16, and torch.heaviside
# requires both args to share a dtype -> RuntimeError. This coerces the scalar to the
# operand's dtype. It is a NO-OP in FP32 (dtypes already match) and leaves the shared
# model file (SNN_models_simpl.py) untouched.
_orig_heaviside = torch.heaviside
def _heaviside_dtype_safe(input, values, *args, **kwargs):
    if torch.is_tensor(values) and values.dtype != input.dtype:
        values = values.to(input.dtype)
    return _orig_heaviside(input, values, *args, **kwargs)
torch.heaviside = _heaviside_dtype_safe

# ----------------------------------------------------------------------------
# Fixed configuration
# ----------------------------------------------------------------------------
PRETRAINED_CKPT_PATH = './results/checkpoints/simplified_stereospike_NoSkip.pth'
TEST_SET_PATH        = './datasets/MVSEC/data/indoor_post/test_set.pt'
LOG_DIR              = './results/logs/accuracy'          # all outputs live here

WARMUP_SAMPLES = 20     # untimed inferences before the measured pass (excluded)
USE_TQDM       = False  # A/B switch: progress bar off to avoid host contention
SEED           = 42     # reproducibility

# Which depth map is the FULL model's reported output.
#   0 -> depth1  = accumulation of prediction layers 4+3+2+1 (full model's real output)  [DEFAULT]
#   3 -> depth4  = single coarsest layer, matches the pruned model (use only for a
#                  "pruned == full's depth4" equivalence check)
FULL_MODEL_DEPTH_INDEX = 0

# The 4 runs to execute. (model_name, is_pruned, precision)
RUN_MATRIX = [
    ("full",   False, "fp32"),
    ("full",   False, "fp16"),
    ("pruned", True,  "fp32"),
    ("pruned", True,  "fp16"),
]


# ----------------------------------------------------------------------------
# Pruned model (identical construction to the original script)
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

# Attach the same static preprocessing method used by the full model
Pruned_Simplified_StereoSpike_NoSkip.reformat_input_data = staticmethod(
    Simplified_StereoSpike_NoSkip.reformat_input_data
)


def load_pretrained_subset(pretrained_sd: dict, model: nn.Module, verbose=False):
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
# Operating-point / environment capture (all READ-ONLY; no sudo needed)
# These make each result file self-documenting and reproducible.
# ----------------------------------------------------------------------------
def _read(path):
    """Read a small sysfs/text file; never raise."""
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def _mhz(path):
    """Read a *_freq sysfs node (Hz) and return MHz; never raise."""
    raw = _read(path)
    try:
        return round(int(raw) / 1e6, 1)
    except Exception:
        return raw

def _run(cmd):
    """Best-effort external command (may need sudo); capture stdout, never raise."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def read_thermal_zones():
    """Return {zone_type: temperature_C} for every thermal zone (GPU-therm is the key one)."""
    zones = {}
    for tz in sorted(glob.glob('/sys/devices/virtual/thermal/thermal_zone*')):
        ztype = _read(os.path.join(tz, 'type'))
        raw   = _read(os.path.join(tz, 'temp'))
        try:
            zones[ztype] = round(int(raw) / 1000.0, 1)   # millideg -> deg C
        except Exception:
            zones[ztype] = raw
    return zones


def read_gpu_freq_mhz():
    """Current GPU frequency in MHz (from devfreq sysfs)."""
    raw = _read('/sys/devices/platform/17000000.gpu/devfreq_dev/cur_freq')
    try:
        return round(int(raw) / 1e6, 1)
    except Exception:
        return raw


def capture_session_manifest(device):
    """Snapshot the operating point + software stack ONCE; written into every run's log."""
    m = {}
    # --- software stack ---
    m['platform']     = platform.platform()
    m['python']       = platform.python_version()
    m['torch']        = torch.__version__
    m['torch_cuda']   = torch.version.cuda
    m['cudnn']        = torch.backends.cudnn.version()
    try:
        import spikingjelly
        m['spikingjelly'] = getattr(spikingjelly, '__version__', 'unknown')
    except Exception as e:
        m['spikingjelly'] = f"<{e}>"
    m['device']       = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    m['l4t_release']  = _read('/etc/nv_tegra_release')
    m['jetpack']      = _run(['bash', '-c',
                              "apt-cache show nvidia-jetpack 2>/dev/null | grep -m1 Version"])
    # --- operating point (as READ from the running system) ---
    m['nvpmodel_status_file'] = _read('/var/lib/nvpmodel/status')   # e.g. 'pmode:0001 fmode:...'
    m['cpu_online']  = _read('/sys/devices/system/cpu/online')
    m['cpu_present'] = _read('/sys/devices/system/cpu/present')
    m['gpu_freq_MHz_at_start'] = read_gpu_freq_mhz()
    # Authoritative dumps (only populated if run with sudo / passwordless sudo):
    m['nvpmodel_q']        = _run(['nvpmodel', '-q'])
    # --- locked clocks read from sysfs (no root needed) ---
    m['gpu_min_MHz'] = _mhz('/sys/devices/platform/17000000.gpu/devfreq_dev/min_freq')
    m['gpu_max_MHz'] = _mhz('/sys/devices/platform/17000000.gpu/devfreq_dev/max_freq')
    m['cpu_cur_kHz'] = {c: _read(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
                        for c in range(4)}          # MODE_15W keeps cores 0-3 online
    m['tpc_pg_mask'] = _read('/sys/devices/platform/gpu.0/tpc_pg_mask')
    m['cudnn_benchmark']   = torch.backends.cudnn.benchmark
    return m


def write_log(log_path, manifest, run_info, summary):
    """Write the human-readable manifest + run summary for one run."""
    def section(title): return f"\n===== {title} =====\n"
    with open(log_path, 'w') as f:
        f.write("StereoSpike latency/MDE test - run log\n")
        f.write(f"written: {datetime.now().isoformat(timespec='seconds')}\n")

        f.write(section("RUN"))
        for k, v in run_info.items():
            f.write(f"{k}: {v}\n")

        f.write(section("OPERATING POINT + STACK (read at session start)"))
        for k, v in manifest.items():
            f.write(f"{k}: {v}\n")

        f.write(section("SUMMARY"))
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


# ----------------------------------------------------------------------------
# Input preparation (kept OUTSIDE the timed region)
# ----------------------------------------------------------------------------
def prepare_input(net, sample, device):
    """Move a dataloader sample to GPU and build the binarized event frame + label."""
    _, wl, wr, tl, tr, label = sample
    wl = wl.to(device, dtype=torch.float); wr = wr.to(device, dtype=torch.float)
    tl = tl.to(device, dtype=torch.float); tr = tr.to(device, dtype=torch.float)
    label = label.to(device)
    _, test_chunks = net.reformat_input_data(wl, wr, tl, tr)
    x = (test_chunks >= 1).to(test_chunks.dtype)     # binary spikes {0., 1.}
    return x, label


def select_prediction(raw_out, is_pruned):
    """Pick the single depth map to score, for pruned vs full model."""
    pred = raw_out[0]                       # net returns (pred, spks)
    if is_pruned:
        return pred                         # pruned: already a single depth tensor
    return pred[FULL_MODEL_DEPTH_INDEX]     # full: pick one map from [d1,d2,d3,d4]


# ----------------------------------------------------------------------------
# One full latency/MDE pass -> writes one CSV + one LOG
# ----------------------------------------------------------------------------
def run_latency_test(net, model_name, is_pruned, precision,
                     test_loader, device, manifest, timestamp):
    use_fp16 = (precision == "fp16")
    stem = f"latency_test_{model_name}_{precision}_{timestamp}"
    csv_path = os.path.join(LOG_DIR, stem + ".csv")
    log_path = os.path.join(LOG_DIR, stem + ".log")

    print(f"\n>>> Run: {model_name} / {precision}  ->  {stem}")
    net.eval()

    # FP16 => autocast (FP16 multiply, FP32 accumulate). FP32 => no-op context.
    def autocast_ctx():
        return (torch.autocast(device_type='cuda', dtype=torch.float16)
                if use_fp16 else contextlib.nullcontext())

    # --- Warm-up: untimed, EXCLUDED from all statistics (stabilizes CUDA/cuDNN) ---
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            x, _ = prepare_input(net, sample, device)
            functional.reset_net(net)               # reset SNN membrane state
            with autocast_ctx():
                _ = net(x)
            if i + 1 >= WARMUP_SAMPLES:
                break
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    # --- Thermal guard: state right before the measured pass ---
    start_temps = read_thermal_zones()
    start_gpu_freq = read_gpu_freq_mhz()

    # --- Measured pass over the WHOLE test set (batch=1, single-stream) ---
    times_ms, mde_list, n_list, rows = [], [], [], []
    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["sample_index", "inference_time_ms", "MDE_m", "valid_pixels_n"])

        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_loader, desc=f"{model_name}/{precision}",
                                                disable=not USE_TQDM)):
                x, label = prepare_input(net, sample, device)   # data prep (not timed)
                functional.reset_net(net)                        # reset (not timed)

                # ---- time ONLY the inference ----
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                with autocast_ctx():
                    raw_out = net(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                dt_ms = (t1 - t0) * 1000.0

                # ---- metric (outside timed region; computed in FP32) ----
                lin_pred = select_prediction(raw_out, is_pruned).float()
                MDE, n = MeanDepthError_n(lin_pred, label)
                mde_v, n_v = float(MDE), float(n)

                rows.append([idx, f"{dt_ms:.6f}", f"{mde_v:.6f}", f"{n_v:.1f}"])  # no I/O in the loop
                times_ms.append(dt_ms); mde_list.append(mde_v); n_list.append(n_v)
        writer.writerows(rows)                                  # written once, after the loop

    # --- Thermal guard: state right after the measured pass ---
    end_temps = read_thermal_zones()
    end_gpu_freq = read_gpu_freq_mhz()

    # --- Aggregate summary ---
    t = np.asarray(times_ms); mde = np.asarray(mde_list); n = np.asarray(n_list)
    total_time_s = t.sum() / 1000.0
    summary = {
        "samples":                 int(t.size),
        "latency_ms_mean":         round(float(t.mean()), 6),
        "latency_ms_std":          round(float(t.std(ddof=1)), 6) if t.size > 1 else 0.0,
        "latency_ms_min":          round(float(t.min()), 6),
        "latency_ms_max":          round(float(t.max()), 6),
        "throughput_sps":          round(float(t.size / total_time_s), 4) if total_time_s > 0 else None,
        # MDE, two ways: simple per-sample mean (with std), and the correct
        # pixel-weighted dataset mean (each sample weighted by its valid-pixel count).
        "MDE_m_mean_of_samples":   round(float(mde.mean()), 6),
        "MDE_m_std_of_samples":    round(float(mde.std(ddof=1)), 6) if mde.size > 1 else 0.0,
        "MDE_m_pixel_weighted":    round(float((mde * n).sum() / n.sum()), 6) if n.sum() > 0 else None,
        "total_valid_pixels":      float(n.sum()),
    }
    run_info = {
        "model": model_name, "pruned": is_pruned, "precision": precision,
        "fp16_method": "torch.autocast (fp16 matmul / fp32 accumulate)" if use_fp16 else "native fp32",
        "full_model_depth_index": FULL_MODEL_DEPTH_INDEX if not is_pruned else "n/a",
        "warmup_samples_excluded": WARMUP_SAMPLES,
        "tqdm_enabled": USE_TQDM, "csv_write": "once after the loop",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "csv_file": os.path.basename(csv_path),
        # thermal guard: confirms the clock held and shows any heating across the pass
        "gpu_freq_MHz_start": start_gpu_freq, "gpu_freq_MHz_end": end_gpu_freq,
        "thermal_C_start": start_temps, "thermal_C_end": end_temps,
    }

    write_log(log_path, manifest, run_info, summary)
    # print(f"    latency {summary['latency_ms_mean']:.3f} +/- {summary['latency_ms_std']:.3f} ms | "
    #       f"MDE(px-weighted) {summary['MDE_m_pixel_weighted']} m | "
    #       f"GPU {start_gpu_freq}->{end_gpu_freq} MHz")
    print(f"    latency {summary['latency_ms_mean']:.3f} +/- {summary['latency_ms_std']:.3f} ms | "
        f"MDE(mean of samples) {summary['MDE_m_mean_of_samples']} m | "
        f"GPU {start_gpu_freq}->{end_gpu_freq} MHz")
    if start_gpu_freq != end_gpu_freq:
        print("    [!] GPU frequency changed during the pass -> possible throttling, check the log.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    # This test targets the Jetson GPU.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: this test must run on the Jetson GPU.")

    # Deterministic kernel selection (no cuDNN autotuning perturbing latency).
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    os.makedirs(LOG_DIR, exist_ok=True)
    device = torch.device('cuda:0')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')   # shared stem for this session

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Session timestamp: {timestamp}")

    # --- Capture operating point + stack ONCE (written into every run's log) ---
    manifest = capture_session_manifest(device)

    # --- Data: whole test set, fixed order, batch = 1 ---
    test_set = torch.load(TEST_SET_PATH, weights_only=False)   # (add weights_only=False if your torch errors here)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                              drop_last=True, pin_memory=True)
    print(f"Test samples: {len(test_loader)}")

    # --- Build FULL model (FP32 weights) and load checkpoint ---
    full = Simplified_StereoSpike_NoSkip(
        input_chans=2 * 2 * 1, tau=3., v_threshold=1.0, v_reset=0.0,
        use_plif=True, multiply_factor=10.,
        surrogate_function=surrogate.ATan(), learnable_biases=False
    ).to(device)
    full.load_state_dict(torch.load(PRETRAINED_CKPT_PATH, map_location=device, weights_only=False))
    full.eval()

    # --- Build PRUNED model from the full one, load matching weights ---
    pruned = Pruned_Simplified_StereoSpike_NoSkip(full).to(device)
    ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location=device, weights_only=False)
    load_pretrained_subset(ckpt.get("state_dict", ckpt), pruned, verbose=True)
    pruned.eval()

    nets = {"full": full, "pruned": pruned}   # FP16 handled by autocast; weights stay FP32

    # --- Execute the 4 runs; each writes its own CSV + LOG ---
    for model_name, is_pruned, precision in RUN_MATRIX:
        run_latency_test(nets[model_name], model_name, is_pruned, precision,
                         test_loader, device, manifest, timestamp)

    print(f"\nDone. Files written to: {LOG_DIR}")


if __name__ == "__main__":
    main()