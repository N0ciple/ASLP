import torch
from torch.optim import SGD

def optimizers_configuration(network_module):
   
    # Get all named parameters
    all_named_params = list(network_module.named_parameters())
    # Identify extraparameters
    mask_params = [param for name, param in all_named_params if 'mask' in name]
    # set of all parameters
    set_all_params = set([param for _, param in all_named_params])
    # Get network parameters only
    set_network_params = set_all_params - set(mask_params)
    # Transform as a list
    network_params = list(set_network_params)

    net_optimizer = SGD(network_params, lr=network_module.hparams.lr, momentum=network_module.hparams.momentum, weight_decay=network_module.hparams.weight_decay)

    return net_optimizer
