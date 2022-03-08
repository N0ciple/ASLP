import torch
import torch.nn as nn
import torch.nn.functional as F


def add_linear_layer_parameters(layer,config={}):
    layer.mask = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)

def add_conv_layer_parameters(layer,coinfig={}):
    layer.mask = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)



def linear_layer_forward(layer,input):
    return F.linear(input, layer.weight * layer.mask, layer.bias)


def conv_layer_forward(layer,input):
    return layer._conv_forward(input, layer.weight * layer.mask)
