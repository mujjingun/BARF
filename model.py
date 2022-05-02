import torch
import torch.nn as nn

class NeRFModel(nn.Module):
    def __init__(self, in_dim=3, in_view_dim=3, skip_connection =[4], out_dim=4, depth=8):
        super(NeRFModel, self).__init__()
        self.depth =depth
        self.in_dim = in_dim
        self.in_view_dim = in_view_dim
        self.skip_connection = skip_connection

        self.linears_before = nn.ModuleList([nn.Linear(in_dim,256)]+[nn.Linear(256,256) if i not in skip_connection else nn.Linear(256+self.in_dim,256) for i in range(self.depth-1)])

        self.linear_density = nn.Linear(256,1)
        self.linear_color = nn.ModuleList([nn.Linear(256,256), nn.Linear(256+self.in_view_dim,128), nn.Linear(128,3)])
    def forward(self, location, direction):
        # TODO: implement positional encoding

        input_pos = location
        inpur_dir = direction
        feature = location
        for i in range(len(self.linears_before)):
            feature = layer(feature) if i not in self.skip_connection else layer(torch.cat([input_pos,feature],-1))

        density = self.linear(density(feature))

        feature = self.linear_color[0](feature)
        feature = self.linear_color[1](torch.cat([feature,input_dir],-1))
        color = self.linear_color[2](feature)

        return density, color
