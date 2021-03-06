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
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    c2w = torch.squeeze(c2w)
    # print(c2w.get_device())

    H, W, f = hwf
    u, v = torch.meshgrid(torch.arange(H), torch.arange(W))

    y = u-(H-1)/2
    x = v-(W-1)/2
    fx = f
    fy = f
    x = x/fx
    y = y/fy

    # o = (torch.Tensor([0,0,0,1])).type(torch.float64)
    o = torch.Tensor([0,0,0,1])
    
    world_o = c2w@torch.unsqueeze(o,-1)
    world_o = world_o[:3]

    camera_d = torch.stack([x,-y,-torch.ones_like(x)],-1)
    # world_d = np.zeros((H,W,3))
    # for h in range(H):
    #     for w in range(W):
    #         world_d[h,w] = c2w[:3,:3]@camera_d[h,w]
    world_d = torch.squeeze(torch.matmul(c2w[:3,:3],camera_d.unsqueeze(-1)))
    world_o = torch.squeeze(world_o)

    return world_o,world_d

def sample_points(world_o,world_d,num_point,device,near=2.,far=6.):
    """
    input:
        world_o   : camera position  / size : (3,)
        world_d   : camera direction / size : (num_ray,3)
        near      : near of view frustum / float
        far       : far of view frustum / float
        num_point : number of sample point of each ray / int
    output:
        points : sampled points / torch Tensor with size (num_ray,num_point,3)
        world_d : camera direction / torch Tensor with size (num_ray, 3)
        lin : z value in camera coordinate of each points / torch Tensor with size (num_ray, 1)
    """
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    num_ray = world_d.shape[0]

    # lin = torch.linspace(near,far,num_point+1,device=device)[:-1] # (num_point,)
    lin = torch.linspace(near,far,num_point+1)[:-1] # (num_point,)
    
    interv = (far-near)/num_point
    # rand = torch.rand((num_ray,num_point,1),device=device)*interv
    rand = torch.rand((num_ray,num_point,1))*interv
    
    lin = torch.unsqueeze(lin.expand(num_ray,num_point),-1)
    lin = lin+rand # z value of camera space / each direction vector is transformed from (x,y,1) in camera space

    world_d_expand = (world_d.expand((num_point,num_ray,3))).transpose(0,1)

    world_o_expand = world_o.expand(world_d_expand.shape)
    points = world_o_expand+world_d_expand*lin


    return points, world_d, lin
