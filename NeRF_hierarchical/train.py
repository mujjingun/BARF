import torch
from sample_points import *
from estimate_color import estimate_color
import numpy as np
from tqdm import tqdm, trange
import imageio as io
import math
import os

mse_loss = lambda output, gt : torch.mean((output-gt)**2)

def train_nerf(model_coarse,model_fine, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device,
               args):

    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(
        params=grad_vars,
        lr=5e-4,
        betas=(0.9,0.999)
    )

    lr_f_start, lr_f_end = 5e-4, 5e-5

    pbar = tqdm(range(args.n_steps))
    for step in pbar:

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
            world_o, selected_d, args.num_points_coarse,device
        )


        #positional encoding
        # sampled_points = pos_encoder.encode(sampled_points,-1)
        # sampled_directions = dir_encoder.encode(sampled_directions,-1)
        #BARF
        # sampled_points = pos_encoder.encode(sampled_points,step)
        # sampled_directions = dir_encoder.encode(sampled_directions,step)

        color, weights = estimate_color(model_coarse, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder)

        h_sampled_points, h_sampled_directions, rev_lin = invtrans_sampling(
            weights, lin, args.num_points_fine, near, far, selected_d, world_o
        )

        color_fine, weights = estimate_color(model_fine, h_sampled_points, h_sampled_directions, rev_lin, pos_encoder, dir_encoder)
        # TODO: compute loss
        optimizer.zero_grad()
        loss = mse_loss(color, gt) + mse_loss(color_fine, gt)
        pbar.set_description(f"loss : {loss:06f} | {loss*1000:02f}")

        loss.backward()
        optimizer.step()

        step_frac = step / args.n_steps

        for param_group in optimizer.param_groups:
            param_group['lr'] = math.exp(
            math.log(lr_f_start) + (math.log(lr_f_end) - math.log(lr_f_start)) * step_frac)

        if (step%1000 == 0) and (step != 0) :
            world_o, world_d = get_rays(hwf,c2w)
            world_d_flatten = world_d.reshape(-1,3)

            selected_d = world_d_flatten

            num_points = 256

            sampled_points, sampled_directions, lin = sample_points(
                world_o, selected_d, num_points, device
            )

            colors = []
            total_pixel = hwf[0]*hwf[1]
            batch_size = 8000
            iter = total_pixel//batch_size
            with torch.no_grad():
                for j in trange(iter):
                    batch_points = sampled_points[batch_size*j:batch_size*(j+1)]
                    batch_directions = sampled_directions[batch_size*j:batch_size*(j+1)]
                    batch_lin = lin[batch_size*j:batch_size*(j+1)]
                    color = estimate_color(model_fine, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder)
                    colors.append(color)

            color = torch.cat(colors,0)

            color = color.reshape(hwf[0],hwf[1],3)
            os.makedirs(f"{args.basedir}/img",exist_ok=True)
            os.makedirs(f"{args.basedir}/ckpt",exist_ok=True)
            io.imsave(f"{args.basedir}/img/{step}.png",color.cpu().numpy())
            torch.save({
                    'step': step,
                    'model_state_dict' : model_fine.state_dict(),
                    'model_coarse_state_dict' : model_coarse.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{args.basedir}/ckpt/{step}.tar")