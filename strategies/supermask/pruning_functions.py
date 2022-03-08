import torch
import importlib
import helper_modules as hm

def prune_net(network, pruning_rate=0):
    with torch.no_grad():
        for layer in hm.layers_with_masks(network):
            layer.mask.copy_(torch.where(layer.mask>=0, torch.ones_like(layer.mask), torch.zeros_like(layer.mask)))
            layer.strategy = importlib.import_module("strategies.standard-with-mask")
