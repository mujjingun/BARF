import numpy as np
import torch


def get_rays(hwf,c2w,K=None):
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

    H, W, f = hwf
    u, v = torch.meshgrid(torch.arange(H), torch.arange(W))

    y = u-(H-1)/2
    x = v-(W-1)/2
    fx = f
    fy = f
    if K is not None:
        fx = K[0][0]
        fy = K[1][1]
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
    # world_d = torch.sum(camera_d[:,:, np.newaxis, :] * c2w[:3,:3], -1)
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


def invtrans_sampling(weights, lin, num_points, near, far, world_d, world_o):
    """
    weights : 
    lin : 
    num_points : 
    near : 
    far :
    world_d :
    world_o :
    """
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    num_rays = weights.shape[0]
    num_sample_points = weights.shape[1]

    nears = near*torch.ones((num_rays,1)) # [num_rays,1]
    lin2 = torch.cat([nears,lin],1) #[num_rays,1+num_points]

    sum_weights = torch.sum(weights,axis=1).unsqueeze(-1).expand(-1,num_sample_points) # [num_rays,num_points]
    zeros = torch.zeros((num_rays,1)) #[num_rays,1]

    cdf = torch.cat([zeros,torch.cumsum(weights,1)/sum_weights],1) #[num_rays,1+num_points]

    rand, _ = torch.sort(torch.rand(1,num_points))

    rand = rand.expand(num_rays,num_points).contiguous()
    w_ind = torch.searchsorted(cdf,rand)

    w_up = torch.gather(cdf,1,w_ind)-rand
    w_down = rand-torch.gather(cdf,1,w_ind-1)

    lin_up = torch.gather(lin2,1,w_ind)
    lin_down = torch.gather(lin2,1,w_ind-1)

    rev_lin = (lin_up*w_down + lin_down*w_up) / (w_up+w_down)


    world_d_expand = world_d.unsqueeze(1).expand((num_rays,num_points,3))
    world_o_expand = world_o.expand(world_d.shape)
    points = world_o_expand+world_d*rev_lin

    return points, world_d_expand, rev_lin
