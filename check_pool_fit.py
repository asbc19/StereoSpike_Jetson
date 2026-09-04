#!/usr/bin/env python3
"""
Check whether the whole test set (860 binarized event frames) fits resident in
GPU memory at once, for the power-test input pool.

It does NOT run the network. It only:
  1) reads free/total VRAM,
  2) builds ONE preloaded frame exactly as the power test will (reformat_input_data
     -> binarize -> keep on GPU),
  3) measures the per-frame VRAM cost by actually holding frames resident,
  4) extrapolates to 860 and prints a verdict with margin.

Run (same env/operating point as the real test):
    python check_pool_fit.py
"""

import torch
from spikingjelly.activation_based import surrogate
from network.SNN_models_simpl import Simplified_StereoSpike_NoSkip

TEST_SET_PATH = './datasets/MVSEC/data/indoor_post/test_set.pt'
N_TOTAL       = 860          # full test set
N_PROBE       = 40           # frames actually loaded to measure real per-frame cost
SAFETY_FRAC   = 0.85         # only call it "fits" if pool <= 85% of free VRAM

device = torch.device('cuda:0')
GiB = 1024 ** 3


def vram_free_total():
    free_b, total_b = torch.cuda.mem_get_info(device)   # bytes, driver-level
    return free_b, total_b


# reuse the model's static preprocessing (no model instance / weights needed)
reformat = Simplified_StereoSpike_NoSkip.reformat_input_data

print(f"Device: {torch.cuda.get_device_name(device)}")
free0, total = vram_free_total()
print(f"VRAM total: {total/GiB:.2f} GiB | free at start: {free0/GiB:.2f} GiB\n")

# --- load dataset (CPU) ---
test_set = torch.load(TEST_SET_PATH, weights_only=False)
loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                     drop_last=True, pin_memory=True)

# --- hold N_PROBE frames resident on GPU and measure the real drop in free VRAM ---
pool = []
free_before, _ = vram_free_total()
one_frame_bytes = None
with torch.no_grad():
    for i, sample in enumerate(loader):
        _, wl, wr, tl, tr, _ = sample
        wl = wl.to(device, dtype=torch.float); wr = wr.to(device, dtype=torch.float)
        tl = tl.to(device, dtype=torch.float); tr = tr.to(device, dtype=torch.float)
        _, test_chunks = reformat(wl, wr, tl, tr)
        x = (test_chunks >= 1).to(test_chunks.dtype)     # binarized frame, stays on GPU
        pool.append(x)
        if one_frame_bytes is None:
            one_frame_bytes = x.element_size() * x.nelement()   # logical tensor size
        if i + 1 >= N_PROBE:
            break
torch.cuda.synchronize(device)

free_after, _ = vram_free_total()
measured_per_frame = (free_before - free_after) / len(pool)      # real allocator cost

print(f"One frame shape:        {tuple(pool[0].shape)}  dtype={pool[0].dtype}")
print(f"Logical tensor size:    {one_frame_bytes/1e6:.3f} MB")
print(f"Measured VRAM / frame:  {measured_per_frame/1e6:.3f} MB  "
      f"(over {len(pool)} resident frames, includes allocator overhead)\n")

# --- extrapolate to the full set ---
est_pool_bytes = measured_per_frame * N_TOTAL
free_now, _ = vram_free_total()
budget = free0 * SAFETY_FRAC

print(f"Estimated pool of {N_TOTAL} frames: {est_pool_bytes/GiB:.2f} GiB")
print(f"Free VRAM at start:               {free0/GiB:.2f} GiB")
print(f"Safety budget ({int(SAFETY_FRAC*100)}% of free):        {budget/GiB:.2f} GiB")

if est_pool_bytes <= budget:
    print("\n==> FITS: preload the whole 860-frame set as the pool.")
else:
    max_frames = int(budget / measured_per_frame)
    print(f"\n==> DOES NOT FIT with margin. A seeded subset of ~{max_frames} frames "
          f"stays within {int(SAFETY_FRAC*100)}% of free VRAM.")

# cleanup
del pool
torch.cuda.empty_cache()