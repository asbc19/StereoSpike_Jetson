'''
Inference Latency Analysis on StereoSpike

2025 - 08 - 11  

Andres Brito

- Use: Simplified Full-Precision without Skip Connections. 
    - Include the pruned model.
- Test on complete test dataset.
- Latency estimation when using CPU and GPU.
    - Average value using test dataset.
    - It tries power estimation for GPU when available.
'''

#################################################################################################################
# Libraries
#################################################################################################################
import os
import copy
import time
import platform
import psutil
import subprocess
import threading
from collections import deque

# Try to import NVML for GPU power readings; optional.
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

#from spikingjelly.clock_driven import functional, surrogate, neuron, layer
from spikingjelly.activation_based import functional, surrogate, neuron, layer

# StereoSpike Network (Model without skip connections --> Tomo's Modification)
from network.SNN_models_simpl import Simplified_StereoSpike_NoSkip, NeuromorphicNet

from network.metrics import MeanDepthError, MeanDepthError_n, OnePixelAccuracy, log_to_lin_depths, disparity_to_depth, depth_to_disparity
from network.loss import Total_Loss

######################################################################################################################
# Initial Parameters
######################################################################################################################
# Set to GPU since we are using normal models
device = device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running on {device}\n')

# For reproducibility
seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Parameters
nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
N_inference = 2
N_warmup = 1
penal = False
penal_beta = 1.
batchsize = 1
learned_metric = 'LIN'
split = '1'
show = False
do_warmup = False

# Dictionary to pfmaps (DW CONV results) and ofmaps before activation function (PW CONV results)
fmaps = {}

# Path to pretrained model
pretrained_ckpt_path = './results/checkpoints/simplified_stereospike_NoSkip.pth'

######################################################################################################################
# Classes
######################################################################################################################
class Pruned_Simplified_StereoSpike_NoSkip(NeuromorphicNet):
    def __init__(self, full_model: nn.Module):
        """
        Build a pruned model by copying the retained modules from a full model instance.
        We copy modules (deepcopy) so the pruned model is independent.
            - Keep: encoder, bottleneck, deconv4, and predict_depth4.
        """
        super().__init__()

        # Deepcopy retained parts to make pruned model independent of full_model
        # Keep same attribute names for easier state_dict mapping
        self.bottom = copy.deepcopy(full_model.bottom)
        self.conv1  = copy.deepcopy(full_model.conv1)
        self.conv2  = copy.deepcopy(full_model.conv2)
        self.conv3  = copy.deepcopy(full_model.conv3)
        self.conv4  = copy.deepcopy(full_model.conv4)

        self.bottleneck = copy.deepcopy(full_model.bottleneck)

        # Keep only deconv4 and predict_depth4
        self.deconv4 = copy.deepcopy(full_model.deconv4)
        self.predict_depth4 = copy.deepcopy(full_model.predict_depth4)

        # # If your original model had extra attributes used externally (e.g. multiply_factor),
        # # copy them as well:
        # if hasattr(full_model, "multiply_factor"):
        #     self.multiply_factor = full_model.multiply_factor

    def forward(self, x):
        """
        Simplified forward that only computes until depth4.
        Matches the dataflow of the original model up to the retained outputs.
        """
        # follow original forward logic but stop early:
        frame = x[:, 0, :, :, :]  # as original

        # Encoder
        out_bottom = torch.heaviside(self.bottom(frame) - 1., torch.tensor(0.0, device=x.device))
        out_conv1 = torch.heaviside(self.conv1(out_bottom) - 1., torch.tensor(0.0, device=x.device))
        out_conv2 = torch.heaviside(self.conv2(out_conv1) - 1., torch.tensor(0.0, device=x.device))
        out_conv3 = torch.heaviside(self.conv3(out_conv2) - 1., torch.tensor(0.0, device=x.device))
        out_conv4 = torch.heaviside(self.conv4(out_conv3) - 1., torch.tensor(0.0, device=x.device))

        # Bottleneck
        out_bottleneck = self.bottleneck(out_conv4)

        # Decoder
        out_deconv4 = torch.heaviside(self.deconv4(out_bottleneck) - 1., torch.tensor(0.0, device=x.device))
        # out_add4 = out_deconv4
        
        # Prediction layer
        depth4 = self.predict_depth4(out_deconv4)

        # Return only the retained outputs
        # The second list is considered as spikes in the loss function, but even there, they are not used.
        return depth4, [out_bottleneck, out_deconv4]

# Attach the static method from original model to the pruned model class
Pruned_Simplified_StereoSpike_NoSkip.reformat_input_data = staticmethod(Simplified_StereoSpike_NoSkip.reformat_input_data)

######################################################################################################################
# Functions
######################################################################################################################

def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate the size of a PyTorch model in megabytes (MB), including quantized models.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        float: Size of the model in megabytes (MB).
    """
    total_size_bytes = 0

    # Iterate over all parameters
    for param in model.parameters():
        if param.is_quantized:
            # For quantized tensors, calculate size based on the underlying data type
            if param.dtype == torch.qint8:
                # qint8 uses 1 byte per element
                total_size_bytes += param.nelement() * 1
            elif param.dtype == torch.quint8:
                # quint8 also uses 1 byte per element
                total_size_bytes += param.nelement() * 1
            elif param.dtype == torch.qint32:
                # qint32 uses 4 bytes per element
                total_size_bytes += param.nelement() * 4
            else:
                raise ValueError(f"Unsupported quantized dtype: {param.dtype}")
        else:
            # For regular tensors, calculate size as usual
            total_size_bytes += param.nelement() * param.element_size()

    # Iterate over all buffers
    for buffer in model.buffers():
        if buffer.is_quantized:
            # For quantized buffers, calculate size based on the underlying data type
            if buffer.dtype == torch.qint8:
                total_size_bytes += buffer.nelement() * 1
            elif buffer.dtype == torch.quint8:
                total_size_bytes += buffer.nelement() * 1
            elif buffer.dtype == torch.qint32:
                total_size_bytes += buffer.nelement() * 4
            else:
                raise ValueError(f"Unsupported quantized dtype: {buffer.dtype}")
        else:
            # For regular buffers, calculate size as usual
            total_size_bytes += buffer.nelement() * buffer.element_size()

    # Convert bytes to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    return total_size_mb

def load_pretrained_subset(pretrained_sd: dict, model: nn.Module, verbose=False):
    """
    Load tensors from pretrained_sd into model when key exists and shape matches.
    Handles optional 'module.' prefix.
    Returns list of copied keys and lists of skipped keys.
    """
    model_sd = model.state_dict()
    copied = []
    skipped = []

    # Normalize pretrained keys (remove 'module.' if present)
    normalized_pre = {}
    for k, v in pretrained_sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        normalized_pre[nk] = v

    for key, tensor in model_sd.items():
        if key in normalized_pre:
            pv = normalized_pre[key]
            if pv.shape == tensor.shape:
                tensor.copy_(pv)
                copied.append(key)
            else:
                skipped.append((key, "shape_mismatch", tensor.shape, pv.shape))
        else:
            skipped.append((key, "missing_in_pretrained"))

    if verbose:
        print(f"\nCopied {len(copied)} tensors into model.")
        if skipped:
            print(f"Skipped {len(skipped)} entries (missing or shape mismatch). Sample:")
            for item in skipped[:10]:
                print(" ", item)

def acc_eval_depth(net, loss_module, data_loader, learned_metric = 'LIN', depthmap = 'depth1'):
    '''
    Evaluate network accuracy as defined by the original authors of StereoSpike
        - Only for binary event frames
        - We can select which depth map is considered the final result (compare accuracies)
            - depth1 --> accumulates the results from prediction 4, 3, 2 and 1 layers
            - depth2 --> accumulates the results from prediction 4, 3 and 2 layers
            - depth3 --> accumulates the results from prediction 4 and 3 layers
            - depth4 --> result from prediction 4 layer

    Arg:
        net: network to evaluate
        loss_module: loss function definition
        data_loader: dataloader with the test dataset
        learned_metric: parameter by default is 'LIN'
        depthmap: parameter to select which depth map is considered the final result

    Returns:
        Print results
    '''
    # Define the depth map selection logic
    # Use a dictionary for mapping
    depth_to_index = {
        'depth1': 0,
        'depth2': 1,
        'depth3': 2,
        'depth4': 3
    }
    
    # Get the index, default to -1 if not found (optional)
    i = depth_to_index.get(depthmap, -1)
    print(f'Depth map {i+1} is selected for evaluation.')
    
    # Initialize values
    running_test_loss = 0
    running_test_MDE = 0
    running_test_n = 0  # To count the number of valid pixels
    running_test_OPA = 0

    with torch.no_grad():
        for sample in tqdm(data_loader):

            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            warmup_chunks, test_chunks = net.reformat_input_data(warmup_chunks_left, warmup_chunks_right,
                                                                test_chunks_left, test_chunks_right)

            functional.reset_net(net)

            # No warmup
            # if do_warmup:
            #     net(warmup_chunks_left, warmup_chunks_right)

            # Apply a binary mask to find all values >= 1 (True/False results)
            # then get back to the original data type.
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            # Inference
            pred, spks = net(test_evframe)     

            # Loss calculation
            loss = loss_module(pred, label, spks)
            net.detach()

            # go to linear depth to calculate MDE
            # select the appropriate prediction based on the depthmap parameter
            if learned_metric == 'LIN':
                lin_pred = pred[i]
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred[i])
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred[i])
            # Get MDE and number of valid pixels from the sample
            MDE, n = MeanDepthError_n(lin_pred, label)

            # go to disparity to calculate 1PA metric
            pred_disp = depth_to_disparity(lin_pred)
            gt_disp = depth_to_disparity(label)

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MDE
            # Include the number if valid pixels
            running_test_n += n
            running_test_OPA += OnePixelAccuracy(pred_disp, gt_disp)
            
    epoch_test_loss = running_test_loss / len(data_loader)
    epoch_test_MDE = running_test_MDE / len(data_loader)
    # Number of valid pixels is averaged over the dataset
    epoch_test_n = running_test_n / len(data_loader)
    epoch_test_OPA = running_test_OPA / len(data_loader)
    test_epoch_summary = "Loss: {}, Mean Depth Error (m): {}, One-Pixel Accuracy: {}".format(
        epoch_test_loss, epoch_test_MDE, epoch_test_OPA)
    print(f'Number of samples tested: {len(data_loader)}')
    print(test_epoch_summary)
    print(f'Average number of valid pixels in the ground truth: {epoch_test_n:0.2f}/{label.numel()}\n')

def acc_eval_pruned(net, loss_module, data_loader, learned_metric = 'LIN'):
    '''
    Evaluate network accuracy as defined by the original authors of StereoSpike
        - Only for binary event frames
        - Consider the pruned model --> only one depth output

    Arg:
        net: network to evaluate
        loss_module: loss function definition
        data_loader: dataloader with the test dataset
        learned_metric: parameter by default is 'LIN'

    Returns:
        Print results
    '''
    # Initialize values
    running_test_loss = 0
    running_test_MDE = 0
    running_test_OPA = 0

    with torch.no_grad():
        for sample in tqdm(data_loader):

            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            warmup_chunks, test_chunks = net.reformat_input_data(warmup_chunks_left, warmup_chunks_right,
                                                                test_chunks_left, test_chunks_right)

            functional.reset_net(net)

            # No warmup
            # if do_warmup:
            #     net(warmup_chunks_left, warmup_chunks_right)

            # Apply a binary mask to find all values >= 1 (True/False results)
            # then get back to the original data type.
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            # Inference (single depth output)
            pred, spks = net(test_evframe)     

            # Loss calculation
            loss = loss_module(pred, label, spks)
            net.detach()

            # go to linear depth to calculate MDE
            if learned_metric == 'LIN':
                lin_pred = pred
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred)
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred)
            MDE = MeanDepthError(lin_pred, label)

            # go to disparity to calculate 1PA metric
            pred_disp = depth_to_disparity(lin_pred)
            gt_disp = depth_to_disparity(label)

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MDE
            running_test_OPA += OnePixelAccuracy(pred_disp, gt_disp)
            
    epoch_test_loss = running_test_loss / len(data_loader)
    epoch_test_MDE = running_test_MDE / len(data_loader)
    epoch_test_OPA = running_test_OPA / len(data_loader)
    test_epoch_summary = "Loss: {}, Mean Depth Error (m): {}, One-Pixel Accuracy: {}\n".format(
        epoch_test_loss, epoch_test_MDE, epoch_test_OPA)
    print(f'Number of samples tested: {len(data_loader)}')
    print(test_epoch_summary)

def _print_hardware_info(device):
    """
        Print summary of CPU / RAM / GPU / torch info.
    """
    
    print("=== Hardware / Environment info ===")
    
    # CPU
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    try:
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
        print(f"CPU physical cores: {cpu_count}, logical: {cpu_count_logical}, freq: {cpu_freq} MHz")
    except Exception:
        pass
    
    # RAM
    try:
        vm = psutil.virtual_memory()
        print(f"RAM total: {vm.total / (1024**3):.2f} GB, available: {vm.available / (1024**3):.2f} GB")
    except Exception:
        pass

    # Torch / CUDA
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available() and device.type == torch.device('cuda').type:
        print(f"CUDA available: True, CUDA visible devices: {torch.cuda.device_count()}")
        print(f"Using device: {torch.cuda.get_device_name(device)} (id={device.index})")
        
        # NVML summary
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                print(f"GPU NVML name: {gpu_name}, memory total: {mem_info.total/(1024**2):.1f} MB, power_limit: {power_limit:.1f} W")
            except Exception as ex:
                print("NVML available but failed to query device info:", ex)
            finally:
                try: pynvml.nvmlShutdown()
                except Exception: pass
    else:
        print("CUDA available: False or device is CPU.")
    print("===================================\n")

def performance_eval(
    net,
    loss_module,
    data_loader,
    learned_metric='LIN',
    pruned=False,
    device=device,
    warmup_batches: int = 5,
    measure_power: bool = True
):
    """
    Run inference over data_loader using the original model evaluation logic,
    and measure latency and (if possible) power.

    Returns a dict with metrics: average_latency_ms, samples_per_sec,
    avg_power_watts (None if not measured), energy_per_sample_joules, total_time_s, total_energy_j.
    """

    # Show the network information
    if pruned:
        print("Pruned model for performance evaluation.")
    else:
        print("Original model for performance evaluation.")

    # Print hardware info
    _print_hardware_info(device)

    # Prepare NVML if requested and available
    nvml_handle = None
    if measure_power and _HAS_NVML and torch.cuda.is_available() and device.type == 'cuda':
        try:
            pynvml.nvmlInit()
            # use device.index if present, else 0
            dev_idx = device.index if device.index is not None else 0
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
            print("NVML initialized for power sampling.")
        except Exception as e:
            print("NVML init failed, skipping power measurements:", e)
            nvml_handle = None
    else:
        if measure_power:
            print("Power measurement requested but NVML unavailable or device not CUDA — power will be skipped.")

    # Prepare model
    net.to(device)
    net.eval()

    # Warmup loop to stabilize GPU / caches
    n_done = 0
    with torch.no_grad():
        for sample in data_loader:
            # same preprocessing as your original function
            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)

            warmup_chunks, test_chunks = net.reformat_input_data(
                warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right
            )
            functional.reset_net(net)
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            # call model once
            _ = net(test_evframe)

            n_done += 1
            if n_done >= warmup_batches:
                break

    # Now main measurement loop
    total_time = 0.0
    total_energy_j = 0.0
    total_samples = 0
    running_test_loss = 0.0
    running_test_MDE = 0.0
    running_test_n = 0
    running_test_OPA = 0.0

    # local helper to get instant power (W) from NVML handle
    def _sample_power_watts(handle):
        try:
            # returns milliwatts
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return float(mw) / 1000.0
        except Exception:
            return None

    # iterate over dataset (measuring)
    with torch.no_grad():
        for sample in tqdm(data_loader):
            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            warmup_chunks, test_chunks = net.reformat_input_data(
                warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right
            )
            functional.reset_net(net)
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            # Batch size is always 1
            batch_size = test_evframe.shape[0] if test_evframe.dim() >= 1 else 1
            # print(f'Batch size: {batch_size}')

            # sample power (before) if possible
            p_before = _sample_power_watts(nvml_handle) if nvml_handle else None

            # start timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            # inference (your original call)
            pred, spks = net(test_evframe)

            # ensure sync for GPU timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            batch_time = t1 - t0

            # sample power (after)
            p_after = _sample_power_watts(nvml_handle) if nvml_handle else None

            # derive batch power estimate (simple average of before/after when available)
            if (p_before is not None) and (p_after is not None):
                batch_power_w = 0.5 * (p_before + p_after)
                batch_energy_j = batch_power_w * batch_time
            else:
                batch_power_w = None
                batch_energy_j = 0.0  # unknown

            # accumulate
            total_time += batch_time
            total_energy_j += batch_energy_j
            total_samples += batch_size

            # Proceed with loss and metrics exactly like your original code
            loss = loss_module(pred, label, spks)
            try:
                net.detach()
            except Exception:
                # Some pruned models may have detach bound differently. Ignore if absent.
                pass
            
            # Select the final depth map (final accumulation) for the original network
            # for the pruned model, we only have one depth output
            if not pruned:
                pred = pred[0]

            # convert predicted depth to linear units according to learned_metric
            if learned_metric == 'LIN':
                lin_pred = pred
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred)
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred)
            else:
                lin_pred = pred

            # Compute metrics (these helper functions assumed present in your codebase)
            MDE, n_valid = MeanDepthError_n(lin_pred, label)
            pred_disp = depth_to_disparity(lin_pred)
            gt_disp = depth_to_disparity(label)

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MDE
            running_test_n += n_valid
            running_test_OPA += OnePixelAccuracy(pred_disp, gt_disp)

    # End of loop: compute summary metrics
    if total_samples == 0:
        raise RuntimeError("No samples processed (data_loader empty?)")

    avg_latency_s = total_time / total_samples
    avg_latency_ms = avg_latency_s * 1000.0
    throughput_sps = total_samples / total_time if total_time > 0 else float('inf')

    avg_power_w = (total_energy_j / total_time) if (total_time > 0 and total_energy_j > 0) else None
    energy_per_sample_j = (total_energy_j / total_samples) if total_samples > 0 and total_energy_j > 0 else None

    # Print results (latency, power/energy, accuracy summary)
    print("\n=== Inference Performance Summary ===")
    print(f"Total samples processed: {total_samples}")
    print(f"Total inference time (s): {total_time:.4f}")
    print(f"Average latency per sample: {avg_latency_ms:.4f} ms")
    print(f"Throughput: {throughput_sps:.2f} samples/s")
    if avg_power_w is not None:
        print(f"Average power during inference: {avg_power_w:.2f} W")
        print(f"Energy per sample: {energy_per_sample_j:.4f} J")
    else:
        print("Power not measured (NVML unavailable or not a CUDA device).")

    # Print accuracy summary (same as your function)
    epoch_test_loss = running_test_loss / len(data_loader)
    epoch_test_MDE = running_test_MDE / len(data_loader)
    epoch_test_n = running_test_n / len(data_loader)
    epoch_test_OPA = running_test_OPA / len(data_loader)
    print(f"\nAccuracy summary: Loss: {epoch_test_loss:.6f}, Mean Depth Error (m): {epoch_test_MDE:.6f}, One-Pixel Accuracy: {epoch_test_OPA:.6f}")
    print(f"Average number of valid pixels in the ground truth: {epoch_test_n:0.2f}/{label.numel()}\n")

    # Clean up NVML
    if nvml_handle is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # Return a dictionary of collected metrics
    return {
        "total_samples": total_samples,
        "total_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_sps": throughput_sps,
        "total_energy_j": total_energy_j,
        "avg_power_w": avg_power_w,
        "energy_per_sample_j": energy_per_sample_j,
        "epoch_test_loss": epoch_test_loss,
        "epoch_test_MDE": epoch_test_MDE,
        "epoch_test_OPA": epoch_test_OPA,
        "avg_valid_pixels": epoch_test_n
    }

def _print_hardware_info_smi(device):
    print("=== Hardware / Environment info ===")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    try:
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        print(f"CPU physical cores: {cpu_count}, logical: {cpu_count_logical}")
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        print(f"RAM total: {vm.total / (1024**3):.2f} GB")
    except Exception:
        pass
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available() and device is not None and device.type == 'cuda':
        print(f"CUDA available: True, devices: {torch.cuda.device_count()}")
        try:
            print(f"Using device: {torch.cuda.get_device_name(device)} (id={device.index})")
        except Exception:
            pass
    else:
        print("CUDA available: False or using CPU device.")
    print("===================================\n")

def _nvidia_smi_sampler(interval_ms: int, samples: list, stop_event: threading.Event):
    """
    Spawn a subprocess 'nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv -lms <interval_ms>'
    and append parsed samples to `samples` as tuples (wall_time, power_w, util_pct).
    Runs until stop_event.is_set().
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,power.draw,utilization.gpu",
        "--format=csv",
        "-lms", str(interval_ms)
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except FileNotFoundError:
        # nvidia-smi not found
        return

    # read header line
    try:
        header = proc.stdout.readline()
    except Exception:
        proc.terminate()
        return

    try:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                # EOF or process ended
                break
            # Example line: '2025/08/11 12:34:56.789, 123.45 W, 12 %\n'
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                # power: strip ' W' if present
                power_str = parts[1].split()[0]
                power_w = float(power_str)
                util_str = parts[2].split()[0]
                util_pct = float(util_str)
                wall_ts = time.time()
                samples.append((wall_ts, power_w, util_pct))
            except Exception:
                continue
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

def _integrate_power_trapezoid(samples, t_start, t_end):
    """
    Compute integrated energy (J) between t_start and t_end using trapezoidal integration
    over the (wall_time, power_w) samples list (assumed sorted by time).
    samples: list of (ts, power_w, util)
    Returns: total_energy_j (float), avg_power_w (float)
    """
    if not samples:
        return 0.0, None

    # Filter samples inside [t_start - margin, t_end + margin]; allow a tiny margin
    filtered = [(ts, p) for (ts, p, u) in samples if (ts >= t_start - 0.5 and ts <= t_end + 0.5)]
    if not filtered:
        # no samples in window; fallback to averaging all samples
        p_vals = [p for (ts, p, u) in samples]
        avg_p = sum(p_vals) / len(p_vals)
        total_energy = avg_p * (t_end - t_start)
        return total_energy, avg_p

    # Ensure sorted
    filtered.sort(key=lambda x: x[0])
    times = [t for (t, p) in filtered]
    powers = [p for (t, p) in filtered]

    # Clip endpoints to t_start, t_end: we assume power constant outside sample times
    total_energy = 0.0
    # integrate by trapezoid between successive samples
    for i in range(len(times) - 1):
        t0, p0 = times[i], powers[i]
        t1, p1 = times[i+1], powers[i+1]
        # interval intersection with [t_start, t_end]
        a = max(t0, t_start)
        b = min(t1, t_end)
        if b <= a:
            continue
        # linear interpolate using trapezoid
        # effective p0' and p1' approximated by p0 and p1 over [a,b]
        total_energy += 0.5 * (p0 + p1) * (b - a)
    # handle before first sample if t_start < first sample time: assume p stays at first sample value
    if t_start < times[0]:
        total_energy += powers[0] * (min(times[0], t_end) - t_start)
    # handle after last sample if t_end > last sample: assume p stays at last sample value
    if t_end > times[-1]:
        total_energy += powers[-1] * (t_end - max(times[-1], t_start))

    avg_power = total_energy / max(1e-12, (t_end - t_start))
    return total_energy, avg_power

def performance_eval_nvidia_smi(
    net,
    loss_module,
    data_loader,
    learned_metric='LIN',
    pruned=False,
    device=device,
    warmup_batches: int = 5,
    sample_interval_ms: int = 5,
    enable_power_sampling: bool = True
):
    """
    Performs evaluation (same logic as performance_eval) while sampling GPU power using nvidia-smi.
    Returns metrics dict including latency and estimated power/energy (if sampling succeeded).
    """
    # Show the network information
    if pruned:
        print("\nPruned model for performance evaluation.")
    else:
        print("\nOriginal model for performance evaluation.")

    # Print hardware info
    _print_hardware_info_smi(device)

    # Prepare sampling data structure & sampler thread
    samples = []           # appended tuples (wall_ts, power_w, util)
    stop_event = threading.Event()
    sampler_thread = None

    has_nvidia_smi = (shutil := __import__('shutil')).which("nvidia-smi") is not None
    can_sample = enable_power_sampling and has_nvidia_smi and torch.cuda.is_available() and device.type == 'cuda'

    if can_sample:
        sampler_thread = threading.Thread(
            target=_nvidia_smi_sampler,
            args=(sample_interval_ms, samples, stop_event),
            daemon=True
        )
        sampler_thread.start()
        print(f"Started nvidia-smi sampler at {sample_interval_ms} ms interval.")
    else:
        print("Power sampling disabled (nvidia-smi not found or device not CUDA).")

    net.to(device)
    net.eval()

    total_time = 0.0
    total_energy = 0.0  # will be computed via samples integration
    total_samples = 0

    # Start overall inference interval markers
    inference_start_wall = None
    inference_end_wall = None

    running_test_loss = 0.0
    running_test_MDE = 0.0
    running_test_n = 0
    running_test_OPA = 0.0

    # Warmup (stabilize GPU kernels)
    with torch.no_grad():
        n_w = 0
        for sample in data_loader:
            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)

            warmup_chunks, test_chunks = net.reformat_input_data(
                warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right
            )
            functional.reset_net(net)
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            _ = net(test_evframe)
            n_w += 1
            if n_w >= warmup_batches:
                break

    # Main evaluation loop
    with torch.no_grad():
        for sample in tqdm(data_loader):
            init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label = sample
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
            test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            warmup_chunks, test_chunks = net.reformat_input_data(
                warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right
            )
            functional.reset_net(net)
            test_evframe = (test_chunks >= 1).to(test_chunks.dtype)

            batch_size = test_evframe.shape[0] if test_evframe.dim() >= 1 else 1

            # accurate start time (synchronize GPU if needed)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            # mark first inference start wall time for sampling integration
            if inference_start_wall is None:
                inference_start_wall = time.time()

            # model inference
            pred, spks = net(test_evframe)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            batch_time = t1 - t0
            total_time += batch_time
            total_samples += batch_size

            # Loss & detach as original
            loss = loss_module(pred, label, spks)
            try:
                net.detach()
            except Exception:
                pass

            # Select the final depth map (final accumulation) for the original network
            # for the pruned model, we only have one depth output
            if not pruned:
                pred = pred[0]

            # convert predicted depth as in your code
            if learned_metric == 'LIN':
                lin_pred = pred
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred)
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred)
            else:
                lin_pred = pred

            MDE, n = MeanDepthError_n(lin_pred, label)
            pred_disp = depth_to_disparity(lin_pred)
            gt_disp = depth_to_disparity(label)

            running_test_loss += loss.item() / test_chunks_left.size(0)
            running_test_MDE += MDE
            running_test_n += n
            running_test_OPA += OnePixelAccuracy(pred_disp, gt_disp)

    inference_end_wall = time.time()

    # Stop sampler and join
    if can_sample and sampler_thread is not None:
        stop_event.set()
        sampler_thread.join(timeout=2.0)
        print(f"Collected {len(samples)} nvidia-smi samples.")

        # integrate power over inference interval
        total_energy, avg_power = _integrate_power_trapezoid(samples, inference_start_wall, inference_end_wall)
    else:
        total_energy, avg_power = 0.0, None

    avg_latency_s = total_time / total_samples if total_samples > 0 else 0.0
    avg_latency_ms = avg_latency_s * 1000.0
    throughput = total_samples / total_time if total_time > 0 else 0.0
    energy_per_sample = (total_energy / total_samples) if (total_samples > 0 and total_energy > 0) else None

    # Print summary
    print("\n=== Inference Performance Summary ===")
    print(f"Total samples processed: {total_samples}")
    print(f"Total inference time (s): {total_time:.4f}")
    print(f"Average latency per sample: {avg_latency_ms:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/s")
    if avg_power is not None:
        print(f"Estimated average GPU power during inference: {avg_power:.2f} W")
        print(f"Estimated total energy consumed (J): {total_energy:.4f} J")
        print(f"Estimated energy per sample (J): {energy_per_sample:.6f} J")
    else:
        print("Power estimation not available (nvidia-smi not found or device not CUDA).")

    # Print accuracy summary (same as your original)
    epoch_test_loss = running_test_loss / len(data_loader)
    epoch_test_MDE = running_test_MDE / len(data_loader)
    epoch_test_n = running_test_n / len(data_loader)
    epoch_test_OPA = running_test_OPA / len(data_loader)
    print(f"\nAccuracy summary: Loss: {epoch_test_loss:.6f}, Mean Depth Error (m): {epoch_test_MDE:.6f}, One-Pixel Accuracy: {epoch_test_OPA:.6f}")
    print(f"Average number of valid pixels in the ground truth: {epoch_test_n:0.2f}/{label.numel()}\n")

    # Return metrics
    return {
        "total_samples": total_samples,
        "total_time_s": total_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_sps": throughput,
        "total_energy_j": total_energy,
        "avg_power_w": avg_power,
        "energy_per_sample_j": energy_per_sample,
        "epoch_test_loss": epoch_test_loss,
        "epoch_test_MDE": epoch_test_MDE,
        "epoch_test_OPA": epoch_test_OPA,
        "avg_valid_pixels": epoch_test_n
    }

#################################################################################################################
# Load Dataset
#################################################################################################################

# Load preprocessed datasets
#test_set = torch.load('./datasets/MVSEC/data/indoor_post/test_set.pt')
test_set = torch.load('./datasets/MVSEC/mini_data/indoor_post/mini_test_set.pt', weights_only=False)

test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=True,
                                               pin_memory=True)

print(f'Number of elements in Test Dataloader: {len(test_data_loader)}\n')

######################################################################################################################
# Network --> Simplified Full-Precision without Skip Connections (Tomo's pretrained model)
######################################################################################################################
# Loss function definition
loss_module = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=penal, beta=penal_beta)

# Instantiate the simplified network without skip connections
net_s_nskip = Simplified_StereoSpike_NoSkip(input_chans=2*2*1, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True, multiply_factor=10., 
                                            surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)

# Load the pretrained model (specify the cpu in this case)
net_s_nskip.load_state_dict(torch.load(pretrained_ckpt_path, map_location=torch.device(device)))
net_s_nskip.eval()

print(net_s_nskip)

# Calculate the size of the model 
model_size_mb = get_model_size_mb(net_s_nskip)
print(f"Simplified Full-precision Model (No skip connections) size: {model_size_mb:.6f} MB")

######################################################################################################################
# Prunned Network --> Simplified Full-Precision without Skip Connections (Tomo's pretrained model)
######################################################################################################################
# Build pruned model from full model (copies modules)
net_p_s_nskip = Pruned_Simplified_StereoSpike_NoSkip(net_s_nskip).to(device)

# Load pretrained checkpoint (state_dict or dict with 'state_dict')
ckpt = torch.load(pretrained_ckpt_path, map_location=device)
pretrained_sd = ckpt.get("state_dict", ckpt)

# Copy weights from pretrained state_dict into pruned model for matching keys/shapes
load_pretrained_subset(pretrained_sd, net_p_s_nskip, verbose=True)

print(net_p_s_nskip)

# Calculate the size of the model 
model_size_mb = get_model_size_mb(net_p_s_nskip)
print(f"Pruned Simplified Full-precision Model (No skip connections) size: {model_size_mb:.6f} MB")

#################################################################################################################
# Accuracy Evaluation
#################################################################################################################
# #--> use final depth map (depth map 1) (accumulating results from prediction 4, 3, 2 and 1 layers)
# acc_eval_depth(net_s_nskip, loss_module, test_data_loader, learned_metric = 'LIN', depthmap = 'depth1')
# print('\n')

#--> use depth map 4 (result from prediction 4 layer)
acc_eval_depth(net_s_nskip, loss_module, test_data_loader, learned_metric = 'LIN', depthmap = 'depth4')
print('\n')

# # --> use pruned model (single depth output). This should give the same results as depth map 4
# acc_eval_pruned(net_p_s_nskip, loss_module, test_data_loader, learned_metric = 'LIN')

#################################################################################################################
# Performance Evaluation
#################################################################################################################
#--> use final depth map (depth map 1) (accumulating results from prediction 4, 3, 2 and 1 layers)
# performance_eval(
#     net=net_s_nskip,
#     loss_module=loss_module,
#     data_loader=test_data_loader,
#     learned_metric='LIN',
#     pruned=False,
#     device=device,
#     warmup_batches = 5,
#     measure_power = True
# )
# Try to estimaye power with NVIDIA-smi
performance_eval_nvidia_smi(
    net=net_s_nskip,
    loss_module=loss_module,
    data_loader=test_data_loader,
    learned_metric='LIN',
    pruned=False,
    device=device,
    warmup_batches = 5,
    sample_interval_ms = 5,
    enable_power_sampling = True
) 

# --> use pruned model (single depth output). This should give the same results as depth map 4 (or close to it)
# performance_eval(
#     net=net_p_s_nskip,
#     loss_module=loss_module,
#     data_loader=test_data_loader,
#     learned_metric='LIN',
#     pruned=True,
#     device=device,
#     warmup_batches = 5,
#     measure_power = True
# )

# Try to estimaye power with NVIDIA-smi
performance_eval_nvidia_smi(
    net=net_p_s_nskip,
    loss_module=loss_module,
    data_loader=test_data_loader,
    learned_metric='LIN',
    pruned=True,
    device=device,
    warmup_batches = 5,
    sample_interval_ms = 2,
    enable_power_sampling = True
) 
