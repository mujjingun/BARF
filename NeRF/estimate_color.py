import torch
import torch.nn.functional as F


def estimate_color(model, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder):
    """
    estimate color based on the NeRF model and the sampled points
    :param model: NeRFModel
    :param sampled_points: tensor shape [num_rays, num_point, 3]
    :param sampled_directions: tensor shape [num_rays, 3]
    :return: (density, color)
             density has shape [num_rays, num_point, 1]
             color has shape [num_rays, 3]
    """
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

    # del z_diff
    
    sampled_directions_normalize = sampled_directions/d_norm
    
    # del d_norm

    
    sampled_directions_normalize = sampled_directions_normalize.unsqueeze(1).expand_as(sampled_points).reshape(-1, 3)
    sampled_points = sampled_points.reshape(-1, 3)

    # print(sampled_points.shape)
    # embed = pos_encoder(sampled_points,-1)
    # print(embed.shape)
    # print("pair 1")
    # print(sampled_points[0])
    # print(embed[0])

    # print("pair 2")
    # print(sampled_points[1])
    # print(embed[1])

    # print("pair 3")
    # print(sampled_points[2])
    # print(embed[2])

    # print("pair 4")
    # print(sampled_points[3])
    # print(embed[3])

    # input("ckpt")

    # density, color = model(
    #     pos_encoder(sampled_points,-1).type(torch.float32),
    #     dir_encoder(sampled_directions_normalize,-1).type(torch.float32)
    # )

    network_input = torch.cat([pos_encoder(sampled_points,-1),dir_encoder(sampled_directions_normalize,-1)],-1)
    raw = model(network_input)
    color = raw[...,:3]
    density = raw[...,3:]


    density = density.reshape(num_rays, num_point)  # density: [num_rays, num_point]
    color = color.reshape(num_rays, num_point, 3)  # color: [num_rays, num_point, 3]

    # ``transparency''
    # delta = torch.unsqueeze(delta,-1)
    # tmp = density * delta
    # tmp = torch.cumsum(tmp, dim=1) - tmp[:, 0].unsqueeze(1)
    # T = torch.exp(-tmp)



    tmp = torch.cat([torch.zeros(density.shape[0],1), density*delta],1)
    tmp = torch.cumsum(tmp,dim=1)[...,:-1]
    T = torch.exp(-tmp)

    alpha = 1. - torch.exp(-density * delta)

    w = T*alpha #[num_rays, num_point]

    color = torch.sum(w.unsqueeze(2) * color, dim=1)
    

    
    return color
