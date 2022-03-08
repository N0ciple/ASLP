import torch
import torch.nn as nn
import torch.nn.functional as F


def add_linear_layer_parameters(layer,config=None):
    layer.mask = nn.Parameter(config['init_value'] * torch.ones_like(layer.weight), requires_grad=True)
    layer.scaling_param = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    if config["signed_constant"]:
        with torch.no_grad():
            layer_weight_std = torch.std(layer.weight)
            layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)
    
def add_conv_layer_parameters(layer,config=None):
    layer.mask = nn.Parameter(config['init_value'] * torch.ones_like(layer.weight), requires_grad=True)
    layer.scaling_param = nn.Parameter(torch.tensor(1.0),requires_grad=True)
    if config["signed_constant"]:
        with torch.no_grad():
            layer_weight_std = torch.std(layer.weight)
            layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)

def linear_layer_forward(layer,input):

    proba_leave = torch.zeros_like(layer.mask)
    
    # Stack probabilities
    log_proba_tensor = torch.stack((layer.mask,proba_leave))

    # # Sample masks according to probabilities
    sampled_tensor = torch.nn.functional.gumbel_softmax(log_proba_tensor, 
                                                        hard=layer.config.get("hard_gumbel",True), 
                                                        tau=layer.config.get("gumbel_tau",1),
                                                        dim=0)
    # Mask the weights to create hat_weight
    if layer.config.get("weight_rescale",False):
        hat_weight = layer.scaling_param * layer.weight * sampled_tensor[0]
    else :
        hat_weight = layer.weight * sampled_tensor[0]

    return F.linear(input, hat_weight, layer.bias)


def conv_layer_forward(layer,input):

    proba_leave = torch.zeros_like(layer.mask)
    
    # # Stack probabilities
    log_proba_tensor = torch.stack((layer.mask,proba_leave))

    # Sample masks according to probabilities
    sampled_tensor = torch.nn.functional.gumbel_softmax(log_proba_tensor, 
                                                        hard=layer.config.get("hard_gumbel",True), 
                                                        tau=layer.config.get("gumbel_tau",1),
                                                        dim=0)
    # Mask the weights to create hat_weight
    if layer.config.get("weight_rescale",False):
        hat_weight = layer.scaling_param * layer.weight * sampled_tensor[0]
    else :
        hat_weight = layer.weight * sampled_tensor[0]

    return layer._conv_forward(input, hat_weight)
