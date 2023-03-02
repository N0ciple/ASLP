import math
import torch
import layer_modules as cl


def compute_preds_one_hot(logits):
    _, idx = torch.max(logits, dim=1)
    return idx


def layers_with_masks(network):
    return [
        layer
        for layer in list(network.modules())
        if isinstance(layer, (cl.PrunableConv2d, cl.PrunableLinear))
    ]
