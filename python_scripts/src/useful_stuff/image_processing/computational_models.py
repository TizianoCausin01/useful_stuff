import os, yaml, sys
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor

sys.path.append("../..")
from useful_stuff.image_processing.utils import read_video, resize_video_array, pool_features, load_model, get_relevant_output_layers, get_activation
from useful_stuff.general_utils.utils import print_wise, decode_matlab_strings, create_RDM, get_module_by_path, get_device

class imgANN():
    def __init__(
        self, 
        model_name: str, 
        pkg: str, 
        img_size: int, 
        relevant_layers=None, 
        pooling='all', 
        weights_type: str = 'DEFAULT',
        dtype=torch.float16,
        attn_implementation: str = None,
        hf_class=None,
        repo_url: str = None,
        revision: str = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.pkg = pkg
        self.img_size = img_size
        self.pooling = pooling
        self.weights_type = weights_type
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.hf_class = hf_class
        self.repo_url = repo_url
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        self.device = get_device(verbose=True)
        self.model = load_model(
            model_name,
            pkg,
            self.device,
            img_size=img_size,
            weights_type=weights_type,
            dtype=dtype,
            attn_implementation=attn_implementation,
            hf_class=hf_class,
            repo_url=repo_url,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.relevant_layers = get_relevant_output_layers(model_name, pkg=pkg) if relevant_layers is None else relevant_layers
        self.features = {}
        self.handles = {}
    # EOF

    def __repr__(self):
        return f"ANN(model={self.model_name}, pkg={self.pkg}, pooling={self.pooling}, device={self.device}, img_size={self.img_size})"
    # EOF

    # --- GETTERS ---
    def get_model_name(self) -> str:
        return self.model_name

    def get_pkg(self) -> str:
        return self.pkg

    def get_img_size(self) -> int:
        return self.img_size

    def get_relevant_layers(self) -> list[str]:
        return self.relevant_layers

    def get_pooling(self) -> str:
        return self.pooling

    def get_device(self):
        return self.device

    def get_model(self):       
        return self.model

    def get_feature_extractor(self):
        return getattr(self, "feature_extractor", None)

    def get_feature_extractor_layers(self):
        return getattr(self, "feature_extractor_layers", None)

    def get_features(self):
        return self.features

    def get_handles(self):
        return self.handles

    # --- SETTERS ---
    def _reload_model(self):
        self.model = load_model(
            self.model_name,
            self.pkg,
            self.device,
            img_size=self.img_size,
            weights_type=self.weights_type,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            hf_class=self.hf_class,
            repo_url=self.repo_url,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )
        if hasattr(self, "feature_extractor"):
            delattr(self, "feature_extractor")
            delattr(self, "feature_extractor_layers")
        self.clear_hooks()

    def set_model_name(self, model_name: str, reload_model: bool = True):
        self.model_name = model_name
        if reload_model:
            self._reload_model()
            self.relevant_layers = get_relevant_output_layers(self.model_name, pkg=self.pkg)
        return self

    def set_pkg(self, pkg: str, reload_model: bool = True):
        self.pkg = pkg
        if reload_model:
            self._reload_model()
            self.relevant_layers = get_relevant_output_layers(self.model_name, pkg=self.pkg)
        return self

    def set_img_size(self, img_size: int, reload_model: bool = True):
        self.img_size = img_size
        if reload_model:
            self._reload_model()
        return self

    def set_relevant_layers(self, relevant_layers: list[str]):
        self.relevant_layers = relevant_layers
        return self

    def set_pooling(self, pooling: str):
        self.pooling = pooling
        return self

    def set_device(self, new_device):
        self.device = new_device
        self.model = self.model.to(self.device)
        if hasattr(self, "feature_extractor"):
            self.feature_extractor = self.feature_extractor.to(self.device)
        return self

    def set_model(self, model):
        self.model = model.to(self.device)
        if hasattr(self, "feature_extractor"):
            delattr(self, "feature_extractor")
            delattr(self, "feature_extractor_layers")
        self.clear_hooks()
        return self

    # --- OTHER METHODS ---
    def get_layer_output_shape(self, layer_name, imsize=224):
        self.create_feature_extractor([layer_name])
        with torch.no_grad():
            in_proxy = torch.randn(1, 3, imsize, imsize).to(self.device)
            tmp_shape = self.feature_extractor(in_proxy)[layer_name].shape[1:]
        self.feature_extractor = None
        return tmp_shape
    # EOF 

    def create_feature_extractor(self, layer_names: list[str] = None):
        if layer_names is None:
            layer_names = self.relevant_layers
        feature_extractor = create_feature_extractor(
            self.model, return_nodes=layer_names
        ).to(self.device) 
        self.feature_extractor = feature_extractor
        self.feature_extractor_layers = layer_names
        return feature_extractor
    # EOF

    def create_forward_hook(self, layer_names: list[str] = None):
        self.clear_hooks()
        if layer_names is None:
            layer_names = self.relevant_layers
        features = {}
        handles = {}
        for l in layer_names:
            module = get_module_by_path(self.model, l)
            h = module.register_forward_hook(get_activation(l, features, self.pooling))
            handles[l] = h
        self.handles = handles
        self.features = features
        return features, handles
    # EOF

    def extract_features(self, x, method="hook"):
        kwargs = x if isinstance(x, dict) else {"x": x}
        with torch.no_grad():
            if method == "hook":
                self.model(**kwargs)
                return self.features
            elif method == "fx":
                return self.feature_extractor(**kwargs)
            else:
                raise ValueError(f"{method=} is not supported")
    # EOF

    def forward(self, x):
        if hasattr(self, "feature_extractor"):
            return self.feature_extractor(x)
        return self.model(x)
    # EOF

    def clear_hooks(self):
        if hasattr(self, "handles"):
            for h in self.handles.values():
                h.remove()
        self.handles = {}
    # EOF
# EOC
