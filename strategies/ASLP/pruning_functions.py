import torch
import importlib
import helper_modules as hm

def prune_net(network, pruning_threshold=0):
    with torch.no_grad():
        for layer in hm.layers_with_masks(network):
            try:
                s_param = layer.scaling_param
            except:
                s_param = 1 # If no scaling params, then set it to 1 (no effect)
            layer.mask.copy_(torch.where(layer.mask>=pruning_threshold, torch.ones_like(layer.mask)*s_param, torch.zeros_like(layer.mask)))
            layer.strategy = importlib.import_module("strategies.standard-with-mask")
        network.strategy = importlib.import_module("strategies.standard-with-mask")
