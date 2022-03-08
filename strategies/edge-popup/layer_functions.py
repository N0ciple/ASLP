
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()
    
class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, sparsity):
        k_val = percentile(mask, sparsity*100)
        return torch.where(mask < k_val, torch.zeros_like(mask), torch.ones_like(mask))

    @staticmethod
    def backward(ctx, g):
        return g, None

def add_linear_layer_parameters(layer,config=None):
    # initialize the mask
    layer.mask = nn.Parameter(torch.Tensor(layer.weight.size()))
    nn.init.kaiming_uniform_(layer.mask, a=math.sqrt(5))

    # NOTE: initialize the weights like this.
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    if config["signed_constant"]:
        with torch.no_grad():

            if config["scale_fan"]:
                print("scaling fans")
                fan = nn.init._calculate_correct_fan(layer.weight, "fan_in")
              
                fan = fan * (1 - config["target_sparsity"])
                gain = nn.init.calculate_gain("relu")
                std = gain / math.sqrt(fan)
                layer.weight.copy_(layer.weight.sign() * std)

            else:
                layer_weight_std = torch.std(layer.weight)
                layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)


    # NOTE: turn the gradient on the weights off
    layer.weight.requires_grad = False
    
def add_conv_layer_parameters(layer,config=None):
    # initialize the mask
    layer.mask = nn.Parameter(torch.Tensor(layer.weight.size()))
    nn.init.kaiming_uniform_(layer.mask, a=math.sqrt(5))

    # NOTE: initialize the weights like this.
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    if config["signed_constant"]:
        with torch.no_grad():
            layer_weight_std = torch.std(layer.weight)
            layer.weight.copy_(torch.sign(layer.weight)*torch.ones_like(layer.weight)*layer_weight_std)

    # NOTE: turn the gradient on the weights off
    layer.weight.requires_grad = False
    


def linear_layer_forward(layer,input):
    subnet = GetSubnet.apply(layer.mask.abs(), layer.config["target_sparsity"])
    w = layer.weight * subnet
    return F.linear(input, w, layer.bias)


def conv_layer_forward(layer,input):
    subnet = GetSubnet.apply(layer.mask.abs(), layer.config["target_sparsity"])
    w = layer.weight * subnet
    return layer._conv_forward(input, w)