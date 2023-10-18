# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False, activation='relu'):
    layers = []
    num_layers = len(mlp_channels)
    
    if activation == 'leaky_relu':
        activation_fn = nn.LeakyReLU
    elif activation == 'gelu':
        activation_fn = nn.GELU
    elif activation == 'elu':
        activation_fn = nn.ELU
    else:
        activation_fn = nn.ReLU

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), activation_fn()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), activation_fn()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)

