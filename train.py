import torch
from sample_points import *
from estimate_color import estimate_color
import numpy as np
from tqdm import tqdm, trange

mse_loss = lambda output, gt : torch.mean((output-gt)**2)

def train_nerf(model, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device,
               args):

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=5e-4, 
        betas=(0.9,0.999),
        weight_decay=args.weight_decay
    )

    pbar = tqdm(range(args.n_steps))
    for step in pbar:
        optimizer.zero_grad()

        # TODO: sample points on ray
        i_train = i_split[0]
        train_idx = np.random.choice(i_train, 1)
        train_im = images[train_idx] # H x W x 3
        c2w = torch.from_numpy(poses[train_idx]).to(device)
        c2w = c2w.type(torch.float32)

        world_o, world_d = get_rays(hwf,c2w) # world_o : (3), world_d (H x W x 3)

        world_o = world_o.to(device)
        world_d = world_d.to(device)

        world_d_flatten = world_d.reshape(-1,3)
        gt_flatten = torch.from_numpy(train_im.reshape(-1,3)).to(device)

        selected_pixel_idx = np.random.choice(np.arange(gt_flatten.shape[0]),args.num_rays)
        selected_d = world_d_flatten[selected_pixel_idx]

        gt = gt_flatten[selected_pixel_idx]

        sampled_points, sampled_directions, lin = sample_points(
            world_o, selected_d, args.num_points,device
        )


        #positional encoding
        # sampled_points = pos_encoder.encode(sampled_points,-1)
        # sampled_directions = dir_encoder.encode(sampled_directions,-1)
        #BARF
        # sampled_points = pos_encoder.encode(sampled_points,step)
        # sampled_directions = dir_encoder.encode(sampled_directions,step)

        color = estimate_color(model, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder)

        # TODO: compute loss
        loss = mse_loss(color, gt)
        pbar.set_description(f"loss : {loss:06f} | {loss*1000:02f}")

        loss.backward()
        optimizer.step()
