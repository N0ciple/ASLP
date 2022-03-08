import torch
from torch.optim import SGD

def optimizers_configuration(network_module):
    all_named_params = list(network_module.named_parameters())
    mask_params = [param for name, param in all_named_params if 'mask' in name]
    mask_optimizer = SGD(mask_params, lr=network_module.hparams.lr, momentum=network_module.hparams.momentum, weight_decay= 0)

    return mask_optimizer

