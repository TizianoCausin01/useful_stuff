import os, yaml, sys
import numpy as np
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

OUTPUT:
    - ipcas: dict[str, IncrementalPCA] -> fitted IncrementalPCA objects for each layer
"""
def compute_img_ipca(ann: imgANN, loader: DataLoader, ipcas: dict[str: IncrementalPCA], device: torch):
    # Forward each image batch and let imgANN hooks collect target-layer features.
    for idx, (images, _) in enumerate(loader):
        images = images.to(device)
        ann.model(images)
        # Update each layer's iPCA with the current batch of activations.
        for layer, features in ann.features.items():
            features = features.detach().cpu().numpy()
            ipcas[layer].partial_fit(features)
        print_wise(f"Computed batch {idx}/{len(loader)} of {ann.model_name}: {list(ipcas.keys())}")
    return ipcas
# EOF 
