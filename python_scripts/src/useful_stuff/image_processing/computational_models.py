import os, yaml, sys
import numpy as np
from einops import reduce, rearrange
import torch, timm
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoModel, AutoConfig
from huggingface_hub import login
#token = os.getenv("HF_TOKEN")
#login(token=token)

sys.path.append("../..")
from useful_stuff.general_utils.utils import print_wise, decode_matlab_strings, create_RDM, get_module_by_path, get_device


def load_torchvision_model(model_name, device, img_size=224, weights_type='DEFAULT'):
    model_cls = getattr(models, model_name) # Get the model class
    weights_name = map_anns_names(model_name)+ '_Weights' # Get the corresponding weights enum, for model_name="alexnet", this gets "AlexNet_Weights"
    weights_enum = getattr(models, weights_name)    
    model = model_cls(weights=getattr(weights_enum, weights_type)).to(device).eval() # Load with DEFAULT weights, by default
    return model
# EOF


def load_timm_model(model_name, device, img_size=384):
    final_name = f"{map_anns_names(model_name, pkg='timm')}_{img_size}" 
    model = timm.create_model(final_name, pretrained=True).to(device)
    return model
# EOF


"""
load_hf_model
Loads a model from the Hugging Face Hub with optional control over dtype, attention implementation, revision, and custom code execution.

INPUT:
    - model_name: str -> name or identifier of the model on the Hugging Face Hub
    - device: str -> device to load the model onto (e.g., 'cpu', 'cuda')
    - dtype: torch.dtype -> data type for model weights (default: torch.float16)
    - attn_implementation: str -> optional attention backend (e.g., 'flash_attention_2')
    - hf_class: class -> Hugging Face model class to use (default: AutoModel)
    - repo_url: str -> optional explicit repository path or URL (overrides model_name)
    - revision: str -> specific model version (branch, tag, or commit hash)
    - trust_remote_code: bool -> whether to allow execution of custom code from the repository

OUTPUT:
    - model: torch.nn.Module -> loaded Hugging Face model on the specified device
"""
def load_hf_model(
    model_name: str,
    device: str,
    dtype=torch.float16,
    attn_implementation: str = None,
    hf_class=None,
    repo_url: str = None,
    revision: str = None,
    trust_remote_code: bool = False
):
    # model source: explicit URL (if provided) or model_name
    source = repo_url if repo_url is not None else model_name
    kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    # end if attn_implementation is not None:
    if revision is not None:
        kwargs["revision"] = revision
    # end if revision is not None:
    cls = hf_class if hf_class is not None else AutoModel
    model = cls.from_pretrained(source, **kwargs).to(device)
    return model
# EOF


"""
load_model
Unified interface to load models from different libraries (timm, torchvision, Hugging Face),
handling package-specific arguments and returning a model on the specified device.

INPUT:
    - model_name: str -> name of the model to load
    - pkg: str -> package source ('timm', 'torchvision', or 'hf')
    - device: str -> device to load the model onto (e.g., 'cpu', 'cuda')
    - img_size: int -> input image size (used for timm and torchvision models)
    - weights_type: str -> weights specification for torchvision models (default: 'DEFAULT')
    - dtype: torch.dtype -> data type for Hugging Face model weights (default: torch.float16)
    - attn_implementation: str -> optional attention backend for Hugging Face models
    - hf_class: class -> Hugging Face model class to use (default: AutoModel)
    - repo_url: str -> optional explicit repository path or URL for Hugging Face models
    - revision: str -> specific model version for Hugging Face models
    - trust_remote_code: bool -> whether to allow execution of custom code (Hugging Face only)

OUTPUT:
    - model: torch.nn.Module -> loaded model on the specified device
"""
def load_model(
    model_name: str,
    pkg: str,
    device: str,
    img_size: int = 384,
    weights_type: str = 'DEFAULT',
    dtype=torch.float16,
    attn_implementation: str = None,
    hf_class=None,
    repo_url: str = None,
    revision: str = None,
    trust_remote_code: bool = False,
):
    if pkg == 'timm':
        model = load_timm_model(model_name, device, img_size=img_size)

    elif pkg == 'torchvision':
        model = load_torchvision_model(
            model_name, device, img_size=img_size, weights_type=weights_type
        )

    elif pkg == 'hf':
        source = repo_url if repo_url is not None else model_name
        kwargs = dict(dtype=dtype, trust_remote_code=trust_remote_code)
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if revision is not None:
            kwargs["revision"] = revision

        cls = hf_class if hf_class is not None else AutoModel
        model = cls.from_pretrained(source, **kwargs).to(device)

    else:
        raise ValueError(f"{pkg=} is not currently supported")

    return model
# EOF


def map_anns_names(model_name, pkg='torchvision'):
    if model_name=='alexnet':
        return 'AlexNet'
    elif model_name== 'resnet50':
        return 'ResNet50'    
    elif model_name== 'resnet18':
        return 'ResNet18'
    elif model_name == 'vit_b_16':
        return 'ViT_B_16'
    elif model_name == 'vit_l_16':
        if pkg=='torchvision':
            return 'ViT_L_16'
        elif pkg=='timm':
            return 'vit_large_patch16'
        # end if pkg=='torchvision':
    elif model_name == 'vgg16':
        return 'VGG16'



"""
get_relevant_output_layers
Returns a list of layer names from a specified deep neural network model
that are approximately aligned with regions in the primate visual cortex
— namely V1, V4, and IT (inferotemporal cortex). These layers are selected
to enable brain-model comparisons or neuroscientific analyses of model representations.

INPUT:
model_name (str): 
    The name of the model architecture. Supported models include:
    - 'resnet18'
    - 'resnet50'
    - 'vgg16'
    - 'alexnet'
    - 'vit_b_16'
OUTPUT:
    List[str]: 
        A list of strings representing layer names in the model. These layers are chosen
        based on their approximate correspondence to stages in the visual processing hierarchy
        (e.g., early visual cortex V1, intermediate V4, and higher-level IT).

Example Usage:
    >>> layers = get_relevant_output_layers('resnet18')
    >>> print(layers)
    ['conv1', 'layer1.0.relu_1', 'layer1.1.relu_1', ..., 'avgpool']

    >>> layers = get_relevant_output_layers('vit_b_16')
    >>> print(layers)
    ['conv_proj', 'encoder.layers.encoder_layer_0.add_1', ..., 'heads.head']
"""
def get_relevant_output_layers(model_name, pkg='torchvision'):
    if model_name == 'resnet18':
        return [
            'conv1',                         
            'layer1.0.relu_1',               
            'layer1.1.relu_1',               
            'layer2.0.relu_1',               
            'layer2.1.relu_1',               
            'layer3.0.relu_1',               
            'layer3.1.relu_1',               
            'layer4.0.relu_1',               
            'layer4.1.relu_1',               
            'avgpool'                        
        ]
    if model_name == 'resnet50':
        return [
            'layer1.0.conv3',
            'layer1.1.conv3',
            'layer1.0.downsample.0', 
            'layer2.0.conv3',
            'layer2.1.conv3',
            'layer2.2.conv3',
            'layer2.3.conv3',
            'layer2.0.downsample.0', 
            'layer3.0.conv3',
            'layer3.1.conv3',
            'layer3.2.conv3',
            'layer3.3.conv3',
            'layer3.4.conv3',
            'layer3.5.conv3',
            'layer3.0.downsample.0', 
            'layer4.0.conv3',
            'layer4.1.conv3',
            'layer4.2.conv3',
            'layer4.0.downsample.0', 
        ]
    if model_name == 'vgg16':
        return [
            'features.0',       # conv1_1 (V1)
            'features.2',       # conv1_2
            'features.5',       # conv2_2
            'features.10',      # conv3_3
            'features.12',      # conv4_1
            'features.16',      # conv4_3
            'features.19',      # conv5_1
            'features.23',      # conv5_3
            'features.30',      # final conv
            'classifier.0'      # first FC layer
        ]
    if model_name == 'alexnet':
        return [
            'features.0',       # conv1
            'features.4',       # conv2
            'features.7',       # conv3
            'features.9',       # conv4
            'features.11',      # conv5
            'classifier.2',     # fc6
            'classifier.5'      # fc7
        ]
    if model_name == 'vit_b_16':
        return [
            'encoder.layers.encoder_layer_0.mlp',
            'encoder.layers.encoder_layer_1.mlp',
            'encoder.layers.encoder_layer_2.mlp',
            'encoder.layers.encoder_layer_3.mlp',
            'encoder.layers.encoder_layer_4.mlp',
            'encoder.layers.encoder_layer_5.mlp',
            'encoder.layers.encoder_layer_6.mlp',
            'encoder.layers.encoder_layer_7.mlp',
            'encoder.layers.encoder_layer_8.mlp',           
            'encoder.layers.encoder_layer_9.mlp',           
            'encoder.layers.encoder_layer_10.mlp',          
            'encoder.layers.encoder_layer_11.mlp',          
            'encoder.layers.encoder_layer_12.mlp',          
            'encoder.layers.encoder_layer_13.mlp',          
            'encoder.layers.encoder_layer_14.mlp',          
            'encoder.layers.encoder_layer_15.mlp',          
            'encoder.layers.encoder_layer_16.mlp',          
            'encoder.layers.encoder_layer_17.mlp',          
            'encoder.layers.encoder_layer_18.mlp',          
            'encoder.layers.encoder_layer_19.mlp',          
            'encoder.layers.encoder_layer_20.mlp',          
            'encoder.layers.encoder_layer_21.mlp',          
            'encoder.layers.encoder_layer_22.mlp',          
            'encoder.layers.encoder_layer_23.mlp',          
        ]
    if model_name == 'vit_l_16':
        if pkg=='torchvision':
            return [
                'encoder.layers.encoder_layer_0.mlp',
                'encoder.layers.encoder_layer_1.mlp',
                'encoder.layers.encoder_layer_2.mlp',
                'encoder.layers.encoder_layer_3.mlp',
                'encoder.layers.encoder_layer_4.mlp',
                'encoder.layers.encoder_layer_5.mlp',
                'encoder.layers.encoder_layer_6.mlp',
                'encoder.layers.encoder_layer_7.mlp',
                'encoder.layers.encoder_layer_8.mlp',           
                'encoder.layers.encoder_layer_9.mlp',           
                'encoder.layers.encoder_layer_10.mlp',          
                'encoder.layers.encoder_layer_11.mlp',          
                'encoder.layers.encoder_layer_12.mlp',          
                'encoder.layers.encoder_layer_13.mlp',          
                'encoder.layers.encoder_layer_14.mlp',          
                'encoder.layers.encoder_layer_15.mlp',          
                'encoder.layers.encoder_layer_16.mlp',          
                'encoder.layers.encoder_layer_17.mlp',          
                'encoder.layers.encoder_layer_18.mlp',          
                'encoder.layers.encoder_layer_19.mlp',          
                'encoder.layers.encoder_layer_20.mlp',          
                'encoder.layers.encoder_layer_21.mlp',          
                'encoder.layers.encoder_layer_22.mlp',          
                'encoder.layers.encoder_layer_23.mlp',          
            ]
        elif pkg=='timm':
            return [
                'blocks.0.mlp.fc2',
                'blocks.1.mlp.fc2',
                'blocks.2.mlp.fc2',
                'blocks.3.mlp.fc2',
                'blocks.4.mlp.fc2',
                'blocks.5.mlp.fc2',
                'blocks.6.mlp.fc2',
                'blocks.7.mlp.fc2',
                'blocks.8.mlp.fc2',
                'blocks.9.mlp.fc2',
                'blocks.10.mlp.fc2',
                'blocks.11.mlp.fc2',
                'blocks.12.mlp.fc2',
                'blocks.13.mlp.fc2',
                'blocks.14.mlp.fc2',
                'blocks.15.mlp.fc2',
                'blocks.16.mlp.fc2',
                'blocks.17.mlp.fc2',
                'blocks.18.mlp.fc2',
                'blocks.19.mlp.fc2',
                'blocks.20.mlp.fc2',
                'blocks.21.mlp.fc2',
                'blocks.22.mlp.fc2',
                'blocks.23.mlp.fc2',
                   ]
    if model_name=='dino_v3_l':
        return [
            "layer.0.mlp.down_proj",
            "layer.1.mlp.down_proj",
            "layer.2.mlp.down_proj",
            "layer.3.mlp.down_proj",
            "layer.4.mlp.down_proj",
            "layer.5.mlp.down_proj",
            "layer.6.mlp.down_proj",
            "layer.7.mlp.down_proj",
            "layer.8.mlp.down_proj",
            "layer.9.mlp.down_proj",
            "layer.10.mlp.down_proj",
            "layer.11.mlp.down_proj",
            "layer.12.mlp.down_proj",
            "layer.13.mlp.down_proj",
            "layer.14.mlp.down_proj",
            "layer.15.mlp.down_proj",
            "layer.16.mlp.down_proj",
            "layer.17.mlp.down_proj",
            "layer.18.mlp.down_proj",
            "layer.19.mlp.down_proj",
            "layer.20.mlp.down_proj",
            "layer.21.mlp.down_proj",
            "layer.22.mlp.down_proj",
            "layer.23.mlp.down_proj",
        ]
    if 'mobilenet_v3_large' in model_name:
        return ["features.6.block.0", "features.15.block.0", "features.6.block.1", "features.15.block.1", "features.6.block.2", "features.15.block.2", "features.6.block.3", "features.15.block.3", "classifier.0", "classifier.3"]
    raise ValueError(f"Model {model_name} not supported in `get_relevant_output_layers()`.")
# EOF


"""
pool_features
Pool the features along spatial or token dimensions depending on the input shape.

INPUT:
    - features: np.ndarray or torch.Tensor -> tensor of shape
        (batch, channels, H, W) for CNNs,
        (batch, tokens, emb_dim) for ViTs,
        or (batch, features) for classifier layers
    - pooling: str -> pooling method to apply ('mean', 'sum', etc.), or 'all' to flatten all non-batch dimensions

OUTPUT:
    - pooled_features: same type as input -> features after applying pooling:
        - CNNs: pooled across H and W
        - ViTs: pooled across tokens
        - classifier layers: unchanged
        - 'all': flattened into (batch, -1)
"""
def pool_features(features, pooling=None):
    dimensions = features.shape
    if pooling is None:
        return features
    if pooling == 'all':
        pooled_features = rearrange(features, 'batch ... -> batch (...)')
        return pooled_features
    if len(dimensions) == 4: # CNNs case
        pooled_features = reduce(features, 'batch_size chan h w -> batch_size chan', pooling)
    elif len(dimensions) == 3: # the ViT case
        pooled_features = reduce(features, 'batch_size tokens emb_dim -> batch_size emb_dim', pooling)
    elif len(dimensions) == 2: # classifier layers, don't apply pooling
        pooled_features = features
    # end
    return pooled_features
# EOF



"""
get_activation
Create a forward hook to extract intermediate features from a module.
The hook optionally pools the output before storing it.

INPUT:
    - name: str -> key under which to store the extracted features
    - features: dict[str: torch.ndarray] -> the features dict ["layer_name": features_array]
    - pooling: str -> pooling method to apply on the output ('all', 'mean', etc.)

OUTPUT:
    - hook: callable -> a function suitable for `module.register_forward_hook`
        that captures the module output in a global `features` dictionary
        under the provided name
"""
def get_activation(name, features, pooling='all'):
    def hook(model, input, output, pooling=pooling):
        output = pool_features(output, pooling)
        features[name] = output
    # EOF
    return hook
# EOF


"""
get_layer_output_shape
Computes the output shape (excluding batch size) of a specific layer 
from a given PyTorch feature extractor when applied to a dummy input 
image of size (1, 3, 224, 224).
INPUT:
- feature_extractor: torch.nn.Module -> A PyTorch model (typically a feature extractor created via torchvision.models.feature_extraction.create_feature_extractor)
                                        which outputs a dictionary of intermediate activations.
            
- layer_name: str -> The name of the layer for which the output shape is desired. This must be one of the keys returned by the feature_extractor.

OUTPUT:
- tmp_shape: Tuple(Int) -> A tuple representing the shape of the output tensor from the specified layer, excluding the batch dimension. For example,
                          (512, 7, 7) for a convolutional layer or (768,) for a transformer block.
            
Example Usage:
    >>> from torchvision.models import resnet18
    >>> from torchvision.models.feature_extraction import create_feature_extractor
    >>> model = resnet18(pretrained=True).eval()
    >>> feat_ext = create_feature_extractor(model, return_nodes=["layer1.0.relu_1"])
    >>> shape = get_layer_out_shape(feat_ext, "layer1.0.relu_1")
    >>> print(shape)
    (64, 56, 56)
"""
def get_layer_output_shape(feature_extractor, layer_name, imsize=224):
    device = get_device() 
    with torch.no_grad():
        in_proxy = torch.randn(1, 3, imsize, imsize).to(device)
        tmp_shape = feature_extractor(in_proxy)[layer_name].shape[1:]
    return tmp_shape
# EOF 

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
    def get_layer_output_shape(self, layer_name):
        module = get_module_by_path(self.model, layer_name)
        f = {}
        with torch.no_grad():
            h = module.register_forward_hook(get_activation(layer_name, f, None))
            in_proxy = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
            self.model(in_proxy)
            tmp_shape = f[layer_name].shape[1:]
        h.remove()
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
