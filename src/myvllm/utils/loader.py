import torch
from torch import nn
import os


def default_weight_loader(module, weight):
    module.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, checkpoint_path: str):
    for file in os.path.listdir(checkpoint_path):
        if file.endswith('.safetensors'):
            file_dir = os.path.join(checkpoint_path, file)
            with safe_open(file_dir, 'pt', device='cpu') as f:
                for weight_name in f.keys():
                    change_name = False
                    for k in packed_module_mapping:
                        if k in weight_name:
                            v, shard_id = packed_module_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight = f.get_tensor(weight_name)
                            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                            weight_loader(param, weight)
                            change_name = True
                            break 
                    if not change_name:
                        param = model.get_parameter(weight_name)
                        weight = f.get_tensor(weight_name)
                        weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                        weight_loader(param, weight)
