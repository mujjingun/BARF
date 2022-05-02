import torch
import torch.nn.functional as F


def estimate_color(model, sampled_points, sampled_directions, lin):
    """
    estimate color based on the NeRF model and the sampled points
    :param model: NeRFModel
    :param sampled_points: tensor shape [num_rays, num_point, 3]
    :param sampled_directions: tensor shape [num_rays, 3]
    :return: (density, color)
             density has shape [num_rays, num_point, 1]
             color has shape [num_rays, 3]
    """
    num_rays, num_point, _ = sampled_points.shape

    # diff = torch.cat([
    #     sampled_points.new_zeros(num_rays, 1, 3),
    #     sampled_points[:, 1:] - sampled_points[:, :-1]
    # ], dim=1)

    # # distance between adjacent samples
    # delta = torch.norm(diff, p=2, dim=2, keepdim=True)

    ####
    lin = torch.squeeze(lin) # num_rays, num_point
    z_diff = lin[:,1:] - lin[:,:-1] # num_rays, num_point-1
    z_diff = torch.cat([z_diff, 1e9*torch.ones((z_diff.shape[0],1))],-1) #num_rays, num_point
    d_norm = torch.norm(sampled_directions,p=2,dim=1, keepdim=True) #num_rays, 1
    delta = z_diff * d_norm

    del z_diff
    del d_norm

    sampled_directions_normalize = sampled_directions/d_norm
    density, color = model(
        sampled_points.reshape(-1, 3),
        sampled_directions_normalize.unsqueeze(1).expand_as(sampled_points).reshape(-1, 3)
    )
    density = density.reshape(num_rays, num_point, 1)  # density: [num_rays, num_point, 1]
    color = color.reshape(num_rays, num_point, 3)  # color: [num_rays, num_point, 3]


    # ``transparency''
    T = torch.exp(-density * delta)

    # integrated pixel color
    color = torch.sum(T * (1 - torch.exp(-density * delta)) * color, dim=1)

    return density, color
