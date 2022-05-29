import math
import torch
from sample_points import *
from estimate_color import estimate_color
import numpy as np
from tqdm import tqdm, trange
import os
from skimage import io
import pytorch3d.transforms

mse_loss = lambda output, gt: torch.mean((output - gt) ** 2)
calc_psnr = lambda mse : -10*torch.log10(mse)

def rot_distance(perturbs):
    R1 = torch.eye(3, device=perturbs.device).unsqueeze(0).expand(perturbs.shape[0], 3, 3)
    R2 = pytorch3d.transforms.so3_exponential_map(perturbs[...,3:])
    
    return pytorch3d.transforms.so3_relative_angle(R1,R2)

def pose_distance(perturbs):
    trans_dist = torch.sqrt((perturbs[:, :3] ** 2).sum(1)).mean(0)
    angle_dist = rot_distance(perturbs).mean(0)

    print(f"pose RMSE: trans = {trans_dist.item():.4f} / "
          f"angle = {math.degrees(angle_dist.item()):.4f}")


def train_nerf(model, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device,
               args):

    num_train = len(i_split[0])
    print(num_train)

    i_train = i_split[0]

    # poses = torch.tensor(poses, dtype=torch.float32, device=device)
    # train_poses = poses[:num_train]
    train_poses = torch.tensor(poses, dtype=torch.float32, device=device)[i_train]
    print(train_poses[0])
    # add perturbations on the camera pose

    # train_poses_t = train_poses[:,:3,3].clone()
    # train_poses_zeros = train_poses[:,3,:3].clone()
    # train_poses[:,3,:3] = train_poses_t
    # train_poses[:,:3,3] = train_poses_zeros
    train_poses = train_poses.transpose(1,2)
    print(train_poses[0])

    pose_params = pytorch3d.transforms.se3_log_map(train_poses)

    print(pytorch3d.transforms.se3_exp_map(torch.Tensor([[1,1,1,1,1,1]])))

    pose_perturbs = torch.nn.Parameter(
        torch.normal(mean=torch.zeros((train_poses.shape[0], 6)), std=torch.ones((train_poses.shape[0],6))*0.15).to(device)
    )
    pose_distance(pose_perturbs)

    lr_f_start, lr_f_end = args.lr_f_start, args.lr_f_end
    lr_f_gamma = (lr_f_end / lr_f_start) ** (1. / args.n_steps)
    lr_p_start, lr_p_end = args.lr_p_start, args.lr_p_end
    lr_p_gamma = (lr_p_end / lr_p_start) ** (1. / args.n_steps)

    optimizer_f = torch.optim.Adam(
        params=[{'params': model.parameters(), 'lr': lr_f_start}],
        betas=(0.9, 0.999),
    )
    optimizer_p = torch.optim.Adam(
        params=[{'params': [pose_perturbs], 'lr': lr_p_start}],
        betas=(0.9, 0.999),
    )

    loss_running_avg = None
    

    pbar = tqdm(range(args.n_steps))
    for step in pbar:
        optimizer_f.zero_grad()
        optimizer_p.zero_grad()

        
        train_idx = np.random.choice(i_train, 1)

        # print("="*100)
        # print("train index is", train_idx)
        # print("(before step) pose perturb at 0 is", pose_perturbs[0])
        # sample points on ray
        

        train_im = images[train_idx]  # H x W x 3

        # print("(before step) pose perturb at train index is", pose_perturbs[train_idx])

        # c2w = poses[train_idx] @ to_matrix(pose_perturbs[train_idx])
        c2w_param = pose_params[train_idx] + pose_perturbs[train_idx]
        c2w = pytorch3d.transforms.se3_exp_map(c2w_param).squeeze().transpose(0,1)
        c2w = c2w.type(torch.float32)

        world_o, world_d = get_rays(hwf, c2w)  # world_o : (3), world_d (H x W x 3)

        world_o = world_o.to(device)
        world_d = world_d.to(device)

        world_d_flatten = world_d.reshape(-1, 3)
        gt_flatten = torch.from_numpy(train_im.reshape(-1, 3)).to(device)

        selected_pixel_idx = np.random.choice(np.arange(gt_flatten.shape[0]), args.num_rays)
        selected_d = world_d_flatten[selected_pixel_idx]

        gt = gt_flatten[selected_pixel_idx]

        sampled_points, sampled_directions, lin = sample_points(
            world_o, selected_d, args.num_points, device
        )

        # positional encoding
        # sampled_points = pos_encoder.encode(sampled_points,-1)
        # sampled_directions = dir_encoder.encode(sampled_directions,-1)
        # BARF
        # sampled_points = pos_encoder.encode(sampled_points,step)
        # sampled_directions = dir_encoder.encode(sampled_directions,step)

        # i_step = step
        # if args.full_pos_enc:
        #     i_step = -1
        color = estimate_color(model, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder,
                               step if args.coarse_to_fine else -1)

        # compute loss
        loss = mse_loss(color, gt)

        if loss_running_avg is None:
            loss_running_avg = loss
        else:
            loss_running_avg = 0.99 * loss_running_avg + 0.01 * loss
        pbar.set_description(f"loss : {loss:06f} | {loss * 1000:02f} ({loss_running_avg * 1000:02f}) "
                             f"lr_f={optimizer_f.param_groups[0]['lr']} lr_p={optimizer_p.param_groups[0]['lr']}")

        loss.backward()
        pose_grad = pose_perturbs.grad[train_idx].clone()
        optimizer_f.step()
        optimizer_p.step()

        # print("(after step) pose perturb at 0 is", pose_perturbs[0])
        # print("(after step)pose perturb at train index is", pose_perturbs[train_idx])
        # print("="*100)

        for grp in optimizer_f.param_groups:
            grp['lr'] *= lr_f_gamma
        for grp in optimizer_p.param_groups:
            grp['lr'] *= lr_p_gamma

        # sanity check and save model
        if (step % 1000 == 0) and (step != 0):
            print(pose_grad)
            pose_distance(pose_perturbs)

            world_o, world_d = get_rays(hwf, c2w)
            world_d_flatten = world_d.reshape(-1, 3)

            selected_d = world_d_flatten

            sampled_points, sampled_directions, lin = sample_points(
                world_o, selected_d, args.num_points, device
            )

            colors = []
            total_pixel = hwf[0] * hwf[1]
            batch_size = 4000
            iter = total_pixel // batch_size
            with torch.no_grad():
                for j in trange(iter):
                    batch_points = sampled_points[batch_size * j:batch_size * (j + 1)]
                    batch_directions = sampled_directions[batch_size * j:batch_size * (j + 1)]
                    batch_lin = lin[batch_size * j:batch_size * (j + 1)]
                    color = estimate_color(model, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder,
                                           step if args.coarse_to_fine else -1)
                    colors.append(color)

            color = torch.cat(colors, 0)

            mse = mse_loss(color, gt_flatten)
            psnr = calc_psnr(mse)
            print(f"PSNR : {psnr.item()}")
            color = color.reshape(hwf[0], hwf[1], 3)

            os.makedirs(f"{args.basedir}/img", exist_ok=True)
            os.makedirs(f"{args.basedir}/ckpt", exist_ok=True)
            io.imsave(f"{args.basedir}/img/{step}.png", color.cpu().numpy())
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_f_state_dict': optimizer_f.state_dict(),
                'optimizer_p_state_dict': optimizer_p.state_dict(),
                'pose': pose_perturbs
            }, f"{args.basedir}/ckpt/{step}.tar")
