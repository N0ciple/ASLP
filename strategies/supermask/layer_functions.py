import torch
import torch.nn as nn
import torch.nn.functional as F


def add_linear_layer_parameters(layer,config=None):
    layer.mask = nn.Parameter(config['init_value'] * torch.ones_like(layer.weight), requires_grad=True)
    if config["signed_constant"]:
        print("# using signed constant")
        with torch.no_grad():
            layer_weight_std = torch.std(layer.weight)
            layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)


def add_conv_layer_parameters(layer,config=None):
    layer.mask = nn.Parameter(config['init_value'] * torch.ones_like(layer.weight), requires_grad=True)
    if config["signed_constant"]:
        print("# using signed constant")
        with torch.no_grad():
            layer_weight_std = torch.std(layer.weight)
            layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)


def linear_layer_forward(layer,input):
    
    sampled_tensor = torch.bernoulli(torch.sigmoid(layer.mask)) + torch.sigmoid(layer.mask) - torch.sigmoid(layer.mask).detach()


    if layer.config.get("weight_rescale",False):
        hat_weight = (layer.weight.numel()/sampled_tensor.sum()) * layer.weight * sampled_tensor
    else :
        hat_weight = layer.weight * sampled_tensor

    
    return F.linear(input, hat_weight, layer.bias)


def conv_layer_forward(layer,input):

    sampled_tensor = torch.bernoulli(torch.sigmoid(layer.mask)) + torch.sigmoid(layer.mask) - torch.sigmoid(layer.mask).detach()

    # Mask the weights to create hat_weight

  
    if layer.config.get("weight_rescale",False) :
        hat_weight = (layer.weight.numel()/sampled_tensor.sum()) * layer.weight * sampled_tensor
    else :
        hat_weight = layer.weight * sampled_tensor



    return layer._conv_forward(input, hat_weight)
