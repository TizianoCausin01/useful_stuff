import os, yaml, sys
import numpy as np
import torch
import time
sys.path.append("../..")
from useful_stuff.general_utils.utils import print_wise
from useful_stuff.image_processing.computational_models import imgANN
"""
compute_img_ipca
Fits one IncrementalPCA object per ANN layer on batches of image activations.
INPUT:
    - ann: imgANN -> model wrapper with hooks already registered for the target layers
    - loader: DataLoader -> batches of transformed images and labels
    - ipcas: dict[str, IncrementalPCA] -> layer-name keyed IncrementalPCA objects
    - device: torch.device | str -> device where image batches are sent for the forward pass
    - rank: int -> the rank of the process
OUTPUT:
    - ipcas: dict[str, IncrementalPCA] -> fitted IncrementalPCA objects for each layer
"""
def compute_img_ipca(ann: imgANN, loader: DataLoader, ipcas: dict[str: IncrementalPCA], device: torch, rank=0, sub_batch_size=None):
    if sub_batch_size is None:
        sub_batch_size = loader.batch_size

    with torch.no_grad():
        # Forward each image batch and let imgANN hooks collect target-layer features.
        for idx, (images, _) in enumerate(loader):
            st_forw = time.time()
            features_list = {layer: [] for layer in ann.handles.keys()}
            batch_size = len(images)
            for i in range(0, batch_size, sub_batch_size):
                sub_batch = images[i:i+sub_batch_size]
                if len(sub_batch) == 0:
                    continue
                sub_batch = sub_batch.to(device)
                ann.model(sub_batch)
                del sub_batch
                for layer_name, features in ann.features.items():
                    f = features.detach().cpu().numpy()
                    features_list[layer_name].append(f)
                    ann.features[layer_name] = None
                    # print_wise(f"sub_batch shape {f.shape}", rank=rank)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache() 
            end_forw = time.time()
            print_wise(f"forward took {end_forw - st_forw}", rank=rank)
            # Update each layer's iPCA with the current batch of activations.
            for layer_name in ann.features.keys():
                st_ipca = time.time()
                f_full = np.concatenate(features_list[layer_name], axis=0)
                print_wise(f"total batch shape {f_full.shape}", rank=rank)
                ipcas[layer_name].partial_fit(f_full)
                features_list[layer_name] = []
                end_ipca = time.time()
                print_wise(f"ipca fit {layer_name} took {end_ipca - st_ipca}", rank=rank)
            print_wise(f"Computed batch {idx}/{len(loader)-1} of {ann.model_name}: {list(ipcas.keys())}", rank=rank)
    return ipcas
# EOF 
