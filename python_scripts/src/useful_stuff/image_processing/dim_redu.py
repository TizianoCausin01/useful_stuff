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
def compute_img_ipca(ann: imgANN, loader: DataLoader, ipcas: dict[str: IncrementalPCA], device: torch, rank=0):
    with torch.no_grad():
        # Forward each image batch and let imgANN hooks collect target-layer features.
        for idx, (images, _) in enumerate(loader):
            st_forw = time.time()
            images = images.to(device)
            ann.model(images)
            end_forw = time.time()
            print_wise(f"forward took {end_forw - st_forw}", rank=rank)
            # Update each layer's iPCA with the current batch of activations.
            for layer, features in ann.features.items():
                st_ipca = time.time()
                features = features.detach().cpu().numpy()
                ipcas[layer].partial_fit(features)
                end_ipca = time.time()
                print_wise(f"ipca fit {layer} took {end_ipca - st_ipca}", rank=rank)
            print_wise(f"Computed batch {idx}/{len(loader)-1} of {ann.model_name}: {list(ipcas.keys())}", rank=rank)
    return ipcas
# EOF 
