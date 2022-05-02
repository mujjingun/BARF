import numpy as np
import torch


def get_rays(hwf,c2w):
    """
    input:
        H : height of image
        W : width of image
        K : camera intrinsic parameter
            K = [fx,  0, cx]
                [ 0, fy, cy]
                [ 0,  0,  1]
        c2w : 3x4 camera to world transformation

    output:
        world_o : numpy array with size (3,)    : camera position in world coordinate
        world_d : numpy array with size (H,W,3) : camera direction in world coordinate
    """
    H, W, f = hwf
    u, v = torch.meshgrid(torch.arange(H), torch.arange(W))
    print(u)
    print(v)
    y = u-(H-1)/2
    x = v-(W-1)/2
    fx = f
    fy = f
    x = x/fx
    y = y/fy

    world_o = c2w@torch.unsqueeze(torch.Tensor([[0,0,0,1]]),-1)
    world_o = world_o[:3]

    camera_d = torch.stack([x,y,torch.ones_like(x)],-1)
    # world_d = np.zeros((H,W,3))
    # for h in range(H):
    #     for w in range(W):
    #         world_d[h,w] = c2w[:3,:3]@camera_d[h,w]
    world_d = torch.sum(camera_d[:,:, np.newaxis, :] * c2w[:3,:3], -1)

    return world_o,world_d

def sample_points(world_o,world_d,num_point,near=2.,far=6.):
    """
    input:
        world_o   : camera position  / size : (3,)
        world_d   : camera direction / size : (num_ray,3)
        near      : near of view frustum / float
        far       : far of view frustum / float
        num_point : number of sample point of each ray / int
    output:
        sampled points : numpy array with size (num_ray,num_point,3)
    """
    num_ray = world_d.shape[0]
    lin = torch.linspace(near,far,num_point) # (num_point,)
    interv = (far-near)/num_point
    rand = torch.rand((num_ray,num_point,1))*interv
    lin = torch.unsqueeze(lin.expand(num_ray,num_point),-1)
    lin = lin+rand

    print(world_d.shape)
    world_d_expand = (world_d.expand((num_point,num_ray,3))).transpose(0,1)
    world_o_expand = world_o.expand(world_d_expand.shape)
    points = world_o_expand+world_d_expand*lin

    return points, world_d
