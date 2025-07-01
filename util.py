# File: util.py
# Description: Utility functions for seed setting, time formatting, data masking, and mixup generation

import os
import time
import random
import torch
import numpy as np

# Ensures reproducibility by setting seeds for all libraries
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Format time duration into a readable string
def format_time(seconds):
    if seconds <= 60:
        return f'{seconds:.1f}s'
    elif seconds <= 3600:
        return f'{int(seconds // 60)}m{seconds % 60:.1f}s'
    else:
        return f'{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m{seconds % 60:.1f}s'

# Drop a percentage of the input tensor's non-zero elements randomly
# Simulates missing data for imputation testing
def DropData(batch_x, d_rate):
    zero_idx = torch.where(batch_x != 0, torch.ones_like(batch_x), torch.zeros_like(batch_x))
    batch_x_nonzero = torch.where(batch_x == 0, torch.full_like(batch_x, -999), batch_x)
    sample_mask = torch.rand(batch_x_nonzero.shape).to(batch_x.device) <= d_rate
    batch_x_drop = torch.where(sample_mask, torch.zeros_like(batch_x_nonzero), batch_x_nonzero)

    final_mask = torch.where(batch_x_drop == 0, torch.ones_like(batch_x_drop), torch.zeros_like(batch_x_drop)) * zero_idx
    final_x = torch.where(batch_x_drop == -999, torch.zeros_like(batch_x_drop), batch_x_drop)
    return final_mask, final_x

# Generates synthetic samples via mixup strategy for contrastive learning
# Mixes only samples with the same label
def mixup_same_label(input_tensor, labels):
    labels_np = labels.detach().cpu().numpy()
    unique_labels = list(set(labels_np))
    mixed_inputs = []

    for _ in range(5):
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels_np) if l == label]
            if len(indices) < 2:
                continue
            i1, i2 = random.sample(indices, 2)
            alpha = random.uniform(0, 1)
            mixed = alpha * input_tensor[i1] + (1 - alpha) * input_tensor[i2]
            mixed_inputs.append((mixed, label))

    if mixed_inputs:
        mixed_data, mixed_labels = zip(*mixed_inputs)
        mixed_data = torch.stack(mixed_data)
        mixed_labels = torch.tensor(mixed_labels, dtype=labels.dtype, device=labels.device)
        return mixed_data, mixed_labels
    else:
        return None, None
