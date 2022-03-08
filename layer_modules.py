import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import importlib


class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features,
                 bias=True, config=None):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 1
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features))

        self.config = config
        self.strategy = importlib.import_module("strategies."+config["strategy"])
        self.strategy.add_linear_layer_parameters(self,config)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu',mode="fan_in")

        if self.bias is not None:
            if "bias_value" in self.config:
                if (type(self.config["bias_value"])==int) or (type(self.config["bias_value"])==float):
                    torch.nn.init.constant_(self.bias, self.config["bias_value"])
                else:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(self.bias, -bound, bound)
            else:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.strategy.linear_layer_forward(self, input)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class PrunableConv2d(torch.nn.modules.conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="same", dilation=1, groups=1,
                 bias=True, padding_mode='zeros', config=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = padding if isinstance(padding, str) else _pair(padding)
        dilation = _pair(dilation)
        
        super(PrunableConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        
        self.config = config

        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu',mode="fan_in")
       
        self.strategy = importlib.import_module("strategies."+config["strategy"])
        self.strategy.add_conv_layer_parameters(self,config)

        if "bias_value" in self.config and bias:
            if (type(self.config["bias_value"])==int) or (type(self.config["bias_value"])==float):
                torch.nn.init.constant_(self.bias, self.config["bias_value"])


    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.strategy.conv_layer_forward(self,input)
