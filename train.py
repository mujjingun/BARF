import math
import torch
from sample_points import *
from estimate_color import estimate_color
import numpy as np
from tqdm import tqdm, trange
import os

mse_loss = lambda output, gt : torch.mean((output-gt)**2)


# From https://learnopencv.com/rotation-matrix-to-euler-angles/
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = theta.new_zeros(theta.shape[0], 3, 3)
    R_x[:, 0, 0] = 1.0
    R_x[:, 1, 1] = torch.cos(theta[:, 0])
    R_x[:, 1, 2] = -torch.sin(theta[:, 0])
    R_x[:, 2, 1] = torch.sin(theta[:, 0])
    R_x[:, 2, 2] = torch.cos(theta[:, 0])

    R_y = theta.new_zeros(theta.shape[0], 3, 3)
    R_y[:, 1, 1] = 1.0
    R_y[:, 0, 0] = torch.cos(theta[:, 1])
    R_y[:, 0, 2] = torch.sin(theta[:, 1])
    R_y[:, 2, 0] = -torch.sin(theta[:, 1])
    R_y[:, 2, 2] = torch.cos(theta[:, 1])

    R_z = theta.new_zeros(theta.shape[0], 3, 3)
    R_z[:, 2, 2] = 1.0
    R_z[:, 0, 0] = torch.cos(theta[:, 2])
    R_z[:, 0, 1] = -torch.sin(theta[:, 2])
    R_z[:, 1, 0] = torch.sin(theta[:, 2])
    R_z[:, 1, 1] = torch.cos(theta[:, 2])

    R = R_z @ R_y @ R_x
    return R


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)

    singular = (sy < 1e-6)
    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    x_sing = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    y_sing = torch.atan2(-R[:, 2, 0], sy)
    z_sing = torch.zeros_like(x_sing)

    x = torch.where(singular, x_sing, x)
    y = torch.where(singular, y_sing, y)
    z = torch.where(singular, z_sing, z)

    return torch.stack([x, y, z], dim=1)


def train_nerf(model, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device,
               args):

    poses = torch.tensor(poses, device=device)
    translation = poses[:, :3, 3]
    U = poses[:, :3, :3]
    angles = rotationMatrixToEulerAngles(U)

    # add perturbations on the camera pose
    translation += torch.randn_like(translation) * 0.26
    angles += torch.randn_like(angles) * math.radians(14.9)

    pose_params = torch.nn.Parameter(torch.cat([translation, angles], dim=1))

    optimizer = torch.optim.Adam(
        params=[
            {'params': model.parameters(), 'lr': 1e-3},
            {'params': [pose_params], 'lr': 3e-3},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    loss_running_avg = None

    pbar = tqdm(range(args.n_steps))
    for step in pbar:
        optimizer.zero_grad()

        # TODO: sample points on ray
        i_train = i_split[0]
        train_idx = np.random.choice(i_train, 1)

        train_im = images[train_idx] # H x W x 3

        translation = pose_params[train_idx, :3]
        angles = pose_params[train_idx, 3:]
        rotation_mat = eulerAnglesToRotationMatrix(angles)
        c2w = rotation_mat.new_zeros(1, 4, 4)
        c2w[0, :3, :3] = rotation_mat
        c2w[0, :3, 3] = translation
        c2w[0, 3, 3] = 1.0

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
        if loss_running_avg is None:
            loss_running_avg = loss
        else:
            loss_running_avg = 0.99 * loss_running_avg + 0.01 * loss
        pbar.set_description(f"loss : {loss:06f} | {loss*1000:02f} ({loss_running_avg*1000:02f})")

        loss.backward()
        optimizer.step()

        step_frac = step / args.n_steps
        optimizer.param_groups[0]['lr'] = 1e-3 + (1e-4 - 1e-3) * step_frac
        optimizer.param_groups[1]['lr'] = 3e-3 + (1e-5 - 3e-3) * step_frac

        #sanity check and save model
        if (step%1000 == 0) and (step != 0) :
            world_o, world_d = get_rays(hwf,c2w)
            world_d_flatten = world_d.reshape(-1,3)

            selected_d = world_d_flatten

            sampled_points, sampled_directions, lin = sample_points(
                world_o, selected_d, args.num_points, device
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
                    color = estimate_color(model, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder)
                    colors.append(color)

            color = torch.cat(colors,0)

            color = color.reshape(hwf[0],hwf[1],3)
            
            os.makedirs(f"{args.basedir}/img",exist_ok=True)
            os.makedirs(f"{args.basedir}/ckpt",exist_ok=True)
            io.imsave(f"{args.basedir}/img/{step}.png",color.cpu().numpy())
            torch.save({
                    'step': step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pose': pose_params
                }, f"{args.basedir}/ckpt/{step}.tar")
