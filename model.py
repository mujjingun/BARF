import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeRFModel(nn.Module):
    def __init__(self, in_dim=3, in_view_dim=3, skip_connection =[4], out_dim=4, depth=8):
        super(NeRFModel, self).__init__()
        self.depth =depth
        self.in_dim = in_dim
        self.in_view_dim = in_view_dim
        self.skip_connection = skip_connection

        self.linear_in = nn.Linear(in_dim,256)
        self.linears_before = nn.ModuleList([nn.Linear(256,256) if i not in skip_connection else nn.Linear(256+self.in_dim,256) for i in range(self.depth-1)])

        self.linear_density = nn.Linear(256,1)
        self.linear_color = nn.ModuleList([nn.Linear(256,256), nn.Linear(256+self.in_view_dim,128), nn.Linear(128,3)])

        self.sigmoid_color = nn.Sigmoid()
        self.softplus_density = nn.Softplus()

    # def forward(self, location, direction):
    #     # TODO: implement positional encoding

    #     input_pos = location
    #     input_dir = direction
    #     feature = nn.ReLU()(self.linear_in(location))
    #     for i in range(len(self.linears_before)):
    #         layer = self.linears_before[i]
    #         feature = layer(feature) if i not in self.skip_connection else layer(torch.cat([input_pos,feature],-1))
    #         feature = nn.ReLU()(feature)

    #     density = self.softplus_density(self.linear_density(feature))

    #     feature = self.linear_color[0](feature)
    #     feature = nn.ReLU()(self.linear_color[1](torch.cat([feature,input_dir],-1)))
    #     color = self.sigmoid_color(self.linear_color[2](feature))

    #     return density, color
    def forward(self, x):
        # TODO: implement positional encoding

        input_pos, input_dir = torch.split(x, [self.in_dim, self.in_view_dim], dim=-1)

        feature = F.relu(self.linear_in(input_pos))
        for i in range(len(self.linears_before)):
            layer = self.linears_before[i]
            feature = layer(feature) if i not in self.skip_connection else layer(torch.cat([input_pos,feature],-1))
            feature = F.relu(feature)

        density = self.softplus_density(self.linear_density(feature))

        feature = F.relu(self.linear_color[0](feature))
        feature = F.relu(self.linear_color[1](torch.cat([feature,input_dir],-1)))
        color = self.sigmoid_color(self.linear_color[2](feature))

        output = torch.cat([color,density],-1)
        return output


class PosEncoding:
    def __init__(self,in_dim,L=0,lower_bound=20000,upper_bound=100000):
        self.in_dim = in_dim
        self.L = L
        self.encode_dim = (2*L+1)*in_dim
        self.pe_fn = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.pe_fn.append(lambda x: x)

        for i in range(L):
            self.pe_fn.append(lambda x, k=i: torch.cos((2**k)*x*math.pi))
            self.pe_fn.append(lambda x, k=i: torch.sin((2**k)*x*math.pi))

    def ret_encode_dim(self):
        return self.encode_dim

    def encode(self, inputs, epoch):
        # inputs: (batch, 3)
        ll = torch.zeros(inputs.shape[0], inputs.shape[1] * self.L * 2, device=inputs.device)

        range_L = torch.arange(self.L, device=inputs.device)
        coeff = math.pi * (2 ** range_L)[None, :, None]  # (1, L, 1)
        cos = torch.cos(coeff * inputs[:, None, :])
        sin = torch.sin(coeff * inputs[:, None, :])  # (batch, L, 3)
        ll_view = ll.view(inputs.shape[0], self.L, 2, inputs.shape[1])  # (batch, L+1, 2, 3)
        ll_view[:, :, 0, :] = cos
        ll_view[:, :, 1, :] = sin

        if (epoch == -1) or (epoch > self.upper_bound): # not use corase-to-fine positional encoding
            return torch.cat([inputs, ll], -1)

        ratio = min(max((epoch - self.lower_bound) / (self.upper_bound - self.lower_bound), 0), 1)
        alpha = ratio * self.L

        weights = (1 - torch.cos(math.pi * torch.clip(alpha - range_L, 0., 1.))) / 2  # (L)
        ll_view[:, :] *= weights[None, :, None, None]

        return torch.cat([inputs, ll], -1)
