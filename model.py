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
        input_dir = direction
        feature = location
        for i in range(len(self.linears_before)):
            layer = self.linears_before[i]
            feature = layer(feature) if i not in self.skip_connection else layer(torch.cat([input_pos,feature],-1))

        density = self.linear(self.linear_density(feature))

        feature = self.linear_color[0](feature)
        feature = self.linear_color[1](torch.cat([feature,input_dir],-1))
        color = self.linear_color[2](feature)

        return density, color


class PosEncoding:
    def __init__(self,in_dim,L=0,upper_bound=2000):
        self.in_dim = in_dim
        self.L = L
        self.encode_dim = (2*L+1)*in_dim
        self.pe_fn = []
        self.upper_bound = upper_bound

        self.pe_fn.append(lambda x:x)


        for i in range(L):
            self.pe_fn.append(lambda x: torch.cos(2**i*x))
            self.pe_fn.append(lambda x: torch.sin(2**i*x))

    def encode(self,inputs,epoch):
        if (epoch == -1) or (epoch > self.upper_bound): # not use corase-to-fine positional encoding
            return torch.cat([fn(inputs) for fn in self.pe_fn], -1)
        else: # use coarse-to-fine positional encoding
            ratio = epoch/self.upper_bound
            alpha = ratio*self.L
            ll = [self.pe_fn[0](inputs)]
            for i in range(self.L):
                cos_fn = self.pe_fn[2*i+1]
                sin_fn = self.pe_fn[2*i+2]
                if alpha >= i+1:
                    ll.append(cos_fn(inputs))
                    ll.append(sin_fn(inputs))
                elif alpha < i:
                    ll.append(torch.zeros_like(inputs))
                    ll.append(torch.zeros_like(inputs))
                else:
                    ll.append(((1-torch.cos(torch.Tensor([alpha-k])))/2)*cos_fn(inputs))
                    ll.append(((1-torch.cos(torch.Tensor([alpha-k])))/2)*sin_fn(inputs))
            return torch.cat(ll, -1)

        


