"""
Initial Test on StereoSpike
2025 - 08 - 07   
Andres Brito
- Verify dataset loading and network inference.
    - Simplified Full-Precision without Skip  Connections (Tomo's result). 
"""

#################################################################################################################
# Libraries
#################################################################################################################
import os
import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import ipywidgets as widgets
from IPython.display import display

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from spikingjelly.clock_driven import functional, surrogate, neuron, layer

# StereoSpike Network (Model without skip connections --> Tomo's Modification)
from network.SNN_models_simpl import Simplified_StereoSpike_NoSkip

from network.metrics import MeanDepthError_original, OnePixelAccuracy, log_to_lin_depths, disparity_to_depth, depth_to_disparity
from network.loss import Total_Loss

#################################################################################################################
# Initial parameters
#################################################################################################################

# Set to GPU since we are using normal models
device = device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Running on {device}\n')

# For reproducibility
torch.manual_seed(42)

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

#################################################################################################################
# Functions
#################################################################################################################

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

def get_nth_sample(dataloader: DataLoader, n: int):
    """
    Get the N-th sample from a PyTorch DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader to retrieve the sample from.
        n (int): The sample index to retrieve.

    Returns:
        The N-th sample (data, target) tuple.
    """
    for i, sample in enumerate(dataloader):
        if i == n:
            return sample  # Return the N-th sample
    raise IndexError("Index out of range in DataLoader.")

def acc_eval(net, loss_module, data_loader, learned_metric = 'LIN'):
    '''
    Evaluate network accuracy as defined by the original authors of StereoSpike
        - Only for binary event frames

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

            # Inference
            pred, spks = net(test_evframe)     

            # Loss calculation
            loss = loss_module(pred, label, spks)
            net.detach()

            # go to linear depth to calculate MDE
            if learned_metric == 'LIN':
                lin_pred = pred[0]
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred[0])
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred[0])
            MDE = MeanDepthError_original(lin_pred, label)

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

#################################################################################################################
# Dataset
#################################################################################################################
"""
- Load the preprocessed test dataset.
    - Mini dataset with a few samples.
"""
cwd = os.getcwd()
print("Current working directory:", cwd)

# Preprocessed dataset
# train_set = torch.load('./datasets/MVSEC/data/indoor_post/train_set.pt')
test_set = torch.load('./datasets/MVSEC/mini_data/indoor_post/mini_test_set.pt', weights_only=False)

# Create corresponding dataloader
# train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
#                                                 batch_size=batchsize,
#                                                 shuffle=True,
#                                                 drop_last=True,
#                                                 pin_memory=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=True,
                                               pin_memory=True)

print(f'Number of elements in Test Dataloader: {len(test_data_loader)}\n')

#################################################################################################################
# Network
#################################################################################################################

# - Use the fixed network: Simplified Full-Precision without Skip Connections:

# Loss function definition
loss_module = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=penal, beta=penal_beta)

# Instantiate the simplified network without skip connections
net_s_nskip = Simplified_StereoSpike_NoSkip(input_chans=2*2*1, tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True, multiply_factor=10., 
                                            surrogate_function=surrogate.ATan(), learnable_biases=False).to(device)

# Load the pretrained model (specify the cpu in this case)
net_s_nskip.load_state_dict(torch.load('./results/checkpoints/simplified_stereospike_NoSkip.pth', map_location=torch.device(device)))
net_s_nskip.eval()

print(net_s_nskip)

# Calculate the size of the model
model_size_mb = get_model_size_mb(net_s_nskip)
print(f"Simplified Full-precision Model (No skip connections) size: {model_size_mb:.6f} MB")

#################################################################################################################
# Evaluation
#################################################################################################################

acc_eval(net_s_nskip, loss_module, test_data_loader, learned_metric = 'LIN')
