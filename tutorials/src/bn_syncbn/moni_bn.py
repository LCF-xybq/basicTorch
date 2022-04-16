import torch
import torch.nn as nn
import torch.nn.modules.batchnorm

def create_inputs():
    return torch.randn(8, 3, 20, 20)

def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x

bn_layer = nn.BatchNorm2d(num_features=3)
inputs = create_inputs()
bn_outputs = bn_layer(inputs)
_, _, expected_out = dummy_bn_forward(inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)
assert torch.allclose(expected_out, bn_outputs)
