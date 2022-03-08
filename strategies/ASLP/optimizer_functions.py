from torch.optim import SGD

def optimizers_configuration(network_module):
    # Get all network parameters
    all_named_params = list(network_module.named_parameters())

    # Get masks and scaling params from all networks parameters
    mask_params = [param for name, param in all_named_params if 'mask' in name]
    scaling_params = [param for name, param in all_named_params if 'scaling_param' in name]

    # Create mask optimizer
    mask_optimizer = SGD(mask_params, lr=network_module.hparams.lr, momentum=network_module.hparams.momentum, weight_decay= network_module.layer_config.get("mask_wd",0))
    
    # Create scaling param optimizer if it exists
    if network_module.hparams.weight_rescale:
        scaling_param_optimizer = SGD(scaling_params, lr=network_module.hparams.sp_lr, momentum=network_module.hparams.momentum, weight_decay= 0)
        return [mask_optimizer,scaling_param_optimizer]
    else:
        return mask_optimizer