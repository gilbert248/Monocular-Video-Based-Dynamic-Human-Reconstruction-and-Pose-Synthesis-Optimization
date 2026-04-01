#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from .activation import SineActivation


act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'sine': SineActivation(omega_0=30),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.shs = nn.Linear(hidden_dim, 16*3)
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        return {'shs': shs, 'opacity': opacity}



class AppearanceDecoder1(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.shs = nn.Linear(hidden_dim, 1*3)  #16 -> 4 ->1
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        return {'shs': shs, 'opacity': opacity}
    


class DeformationDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, weight_norm=True, act='gelu', disable_posedirs=False):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.sine = SineActivation(omega_0=30)
        self.disable_posedirs = disable_posedirs
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
        self.skinning = nn.Linear(hidden_dim, 24)
        
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
            # nn.init.kaiming_normal_(self.skinning_linear.weight, 0.0, nonlinearity='relu')
            
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        if not disable_posedirs:
            self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
            torch.nn.init.constant_(self.blendshapes.bias, 0.0)
            torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        
    def forward(self, x):
        x = self.net(x)
        if not self.disable_posedirs:
            posedirs = self.blendshapes(x)
            posedirs = posedirs.reshape(207, -1)
            
        lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
        lbs_weights = F.gelu(lbs_weights)
        
        return {
            'lbs_weights': lbs_weights,
            'posedirs': posedirs if not self.disable_posedirs else None,
        }

# from torch import nn
# class DeformationDecoder(nn.Module):
#     def __init__(self, n_features=96, hidden_dim=128,  disable_posedirs=False):
#         super().__init__()
#         self.n_features = n_features
#         self.hidden_dim = hidden_dim
#         self.net_init = nn.Sequential(
#             nn.Linear(self.n_features, self.hidden_dim),
#             nn.GELU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.GELU()
#         )
#         self.lbs_net1 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         # nn.init.kaiming_normal_(self.lbs_net1.weight, 0.0, nonlinearity='relu')
#         self.lbs_net1 = nn.utils.weight_norm(self.lbs_net1)
#         self.lbs_net2 = nn.Linear(self.hidden_dim, 24)

#         self.posedir_net = nn.Linear(self.hidden_dim, 3*207)
#         nn.init.kaiming_normal_(self.posedir_net.weight, 0.0, nonlinearity='relu')

#     def forward(self, x):
#         x = self.net_init(x)
#         x_lbs = F.gelu(self.lbs_net1(x))
#         x_lbs = F.gelu(self.lbs_net2(x_lbs))
#         # x_lbs = F.softmax(x_lbs * 10.0, dim=1)
#         x_posedir = self.posedir_net(x)
#         x_posedir = x_posedir.reshape(207, -1)
#         return {'lbs_weights': x_lbs, 'posedirs': x_posedir}


class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 2 if use_surface else 3))
        
    def forward(self, x):
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = F.gelu(self.scales(x))
                
        return {
            'xyz': xyz,
            'rotations': rotations,
            'scales': scales,
        }


class GeometryDecoder1(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 3))
        # self.rotations = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 6))
        self.scales = nn.Sequential(self.net, nn.Linear(self.hidden_dim, 1))
        
    def forward(self, x):
        xyz = self.xyz(x)
        # rotations = self.rotations(x)
        scales = F.gelu(self.scales(x))
                
        return {
            'xyz': xyz,
            # 'rotations': rotations,
            'scales': scales,
        }



class GeometryDecoder2(torch.nn.Module):
    def __init__(self, n_features, use_surface=False, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.xyz = nn.Linear(self.hidden_dim, 3)
        self.scales = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, x):
        xyz = self.xyz(x)
        scales = F.gelu(self.scales(x))
                
        return {
            'xyz': xyz,
            'scales': scales,
        }



class AppearanceSemanticDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu', N_cls=23):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.shs = nn.Linear(hidden_dim, 16*3)
        self.semantic = nn.Sequential(nn.Linear(self.hidden_dim, N_cls, nn.Softmax()))
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        semantic = self.semantic(x)
        return {'shs': shs, 'opacity': opacity, 'semantic': semantic}