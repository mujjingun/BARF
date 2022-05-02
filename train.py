import torch
from sample_points import sample_points
from estimate_color import estimate_color
import numpy as np

mse_loss = lambda output, gt : torch.mean((output-gt)**2)

def train_nerf(model,
               images, poses, render_poses, hwf, i_split,
               args):

    optimizer = torch.optim.Adam(
        model.parameters(),
        5e-4, args.weight_decay
    )
    
    for step in range(args.n_steps):
        optimizer.zero_grad()

        # TODO: sample points on ray
        i_train = i_split[0]
        train_idx = np.random.choice(i_train, 1)
        train_im = images[train_idx] # H x W x 3
        c2w = poses[train_idx]

        world_o, world_d = get_rays(hwf,c2w) # world_o : (3), world_d (H x W x 3)

        world_d_flatten = world_d.reshape(-1,3)
        gt_flatten = train_im.reshape(-1,3)

        selected_pixel_idx = np.random.choice(torch.arange(gt_flatten.shape[0]),args.num_rays)
        selected_d = world_d_flatten[selected_pixel_idx]

        gt = gt_flatten[selected_pixel_idx]


        sampled_points, sampled_directions, lin = sample_points(
            selected_d, world_o, args.num_points
        )

        color = estimate_color(model, sampled_points, sampled_directions, lin)

        # TODO: compute loss
        loss = mse_loss(color, gt)

        loss.backward()
        optimizer.step()
