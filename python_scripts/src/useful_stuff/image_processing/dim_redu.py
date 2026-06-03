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
def compute_img_ipca(ann: imgANN, loader: DataLoader, ipcas: dict[str: IncrementalPCA], device: torch, rank=0, sub_batches=None):
    if sub_batch_size is None:
        sub_batch_size = loader_batch_size

    features_list = {layer: [] for layer in ann.handles.keys()}
    with torch.no_grad():
        # Forward each image batch and let imgANN hooks collect target-layer features.
        for idx, (images, _) in enumerate(loader):
            st_forw = time.time()
            for i in range(0, batch_size, sub_batch_size):
                sub_batch = batch[i:i+sub_batch_size]
                sub_batch = sub_batch.to(device)
                ann.model(sub_batch)
                for layer, features in ann.features.items():
                    f = features.detach().cpu().numpy()
                    features_list[layer].append(f)
                    ANN.features[layer_name] = None
                    torch.cuda.empty_cache() 
                        
            end_forw = time.time()
            print_wise(f"forward took {end_forw - st_forw}", rank=rank)
            # Update each layer's iPCA with the current batch of activations.
            for layer in ann.features.keys():
                st_ipca = time.time()
                ipcas[layer].partial_fit(features_list[layer_name])
                features_list[layer_name] = []
                end_ipca = time.time()
                print_wise(f"ipca fit {layer} took {end_ipca - st_ipca}", rank=rank)
            print_wise(f"Computed batch {idx}/{len(loader)-1} of {ann.model_name}: {list(ipcas.keys())}", rank=rank)
    return ipcas
# EOF 
