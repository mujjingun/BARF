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


def procrustes_analysis(X0, X1):  # [N, 3]
    # translation
    mu0 = X0.mean(dim=0, keepdim=True)
    mu1 = X1.mean(dim=0, keepdim=True)
    X0 = X0 - mu0
    X1 = X1 - mu1

    # scale
    s0 = (X0**2).sum(dim=-1).mean().sqrt()  # TODO replace with std
    s1 = (X1**2).sum(dim=-1).mean().sqrt()
    X0 = X0 / s0
    X1 = X1 / s1

    # obtain rotation matrix
    U, S, Vh = np.linalg.svd((X0.T @ X1).double().cpu().numpy())
    R = U @ Vh
    if np.linalg.det(R) < 0:
        R[2] *= -1
    R = torch.tensor(R, device=X0.device, dtype=torch.float32)
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    return mu0[0], mu1[0], s0, s1, R


@torch.no_grad()
def pose_distance(truth, perturbs):
    # obtain camera positions
    truth, perturbs = invert(truth), invert(perturbs)  # (N, 3, 4)

    origin = torch.tensor([[0., 0., 0., 1.]], device=truth.device).unsqueeze(-1)
    truth_origin = (truth @ origin).squeeze(2)
    perturbs_origin = (perturbs @ origin).squeeze(2)  # (N, 3)

    # perform procrustes analysis
    truth_mu, perturbs_mu, truth_scale, perturbs_scale, rotation = procrustes_analysis(truth_origin, perturbs_origin)

    # align the perturbed origin to ground truth
    aligned_origin = ((perturbs_origin - perturbs_mu) / perturbs_scale) @ rotation.T * truth_scale + truth_mu  # (N, 3)
    R_aligned = perturbs[..., :3] @ rotation.T  # (N, 3, 3)

    trans_dist = (truth_origin - aligned_origin).norm(dim=-1).mean()
    angle_dist = pytorch3d.transforms.so3_relative_angle(truth[..., :3], R_aligned).mean(0)

    print(f"pose error: trans = {trans_dist.item():.4f} / "
          f"angle = {math.degrees(angle_dist.item()):.4f}")


def to_matrix(R, t):
    R = R.float()
    t = t.float()
    pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
    return pose


def invert(pose):
    # invert a camera pose
    R, t = pose[..., :3], pose[..., 3:]
    R_inv = R.transpose(-1, -2)  # TODO replace with .mT
    t_inv = (-R_inv @ t)[..., 0]
    pose_inv = to_matrix(R=R_inv, t=t_inv)
    return pose_inv


def expand(M):
    return torch.cat([
        M,
        torch.tensor([[[0., 0., 0., 1.]]], device=M.device).expand(M.shape[0], -1, -1)
    ], dim=1)


def train_nerf(model, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device, near, far,
               args):

    num_train = len(i_split[0])
    print(num_train)

    i_train = i_split[0]

    # poses = torch.tensor(poses, dtype=torch.float32, device=device)
    # train_poses = poses[:num_train]
    train_poses = torch.tensor(poses, dtype=torch.float32, device=device)[i_train]
    # add perturbations on the camera pose

    # train_poses_t = train_poses[:,:3,3].clone()
    # train_poses_zeros = train_poses[:,3,:3].clone()
    # train_poses[:,3,:3] = train_poses_t
    # train_poses[:,:3,3] = train_poses_zeros
    train_poses = invert(train_poses[:, :3, :])

    pose_noise = torch.normal(
        mean=torch.zeros((train_poses.shape[0], 6)),
        std=torch.ones((train_poses.shape[0], 6)) * 0.15).to(device)

    pose_perturbs = torch.nn.Parameter(
        torch.zeros((train_poses.shape[0], 6), device=device)
    )
    calc_poses = (train_poses @
                  pytorch3d.transforms.se3_exp_map(pose_noise).mT @
                  pytorch3d.transforms.se3_exp_map(pose_perturbs).mT)
    pose_distance(train_poses, calc_poses)

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
    

    pbar = tqdm(range(args.n_steps + 1))
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
        c2w = (train_poses[train_idx] @
               pytorch3d.transforms.se3_exp_map(pose_noise)[train_idx].mT @
               pytorch3d.transforms.se3_exp_map(pose_perturbs)[train_idx].mT)
        c2w = invert(c2w)
        c2w = c2w.type(torch.float32)

        world_o, world_d = get_rays(hwf, c2w)  # world_o : (3), world_d (H x W x 3)

        world_o = world_o.to(device)
        world_d = world_d.to(device)

        world_d_flatten = world_d.reshape(-1, 3)
        gt_flatten = torch.from_numpy(train_im.reshape(-1, 3)).to(device)

        selected_pixel_idx = np.random.choice(np.arange(gt_flatten.shape[0]), args.num_rays, replace=False)
        selected_d = world_d_flatten[selected_pixel_idx]

        gt = gt_flatten[selected_pixel_idx]

        sampled_points, sampled_directions, lin = sample_points(
            world_o, selected_d, args.num_points, device, near, far
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
        color, depth = estimate_color(model, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder,
                               step if args.coarse_to_fine else -1, args.white_background)

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
            calc_poses = (train_poses @
                          pytorch3d.transforms.se3_exp_map(pose_noise).mT @
                          pytorch3d.transforms.se3_exp_map(pose_perturbs).mT)
            pose_distance(train_poses, calc_poses)

        if (step % 8000 == 0) and (step != 0):
            world_o, world_d = get_rays(hwf, c2w)
            world_d_flatten = world_d.reshape(-1, 3)

            selected_d = world_d_flatten

            colors = []
            depths = []
            total_pixel = hwf[0] * hwf[1]
            batch_size = 8000 // 8
            iter = total_pixel // batch_size
            with torch.no_grad():
                for j in trange(iter):
                    batch_points, batch_directions, batch_lin = sample_points(
                        world_o, selected_d[batch_size * j:batch_size * (j + 1)], args.num_points*8, device
                    )
                    color, depth = estimate_color(model, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder,
                                           step if args.coarse_to_fine else -1, white_background)
                    colors.append(color)
                    depths.append(depth)

            color = torch.cat(colors, 0)
            depth = torch.cat(depths, 0)

            mse = mse_loss(color, gt_flatten)
            psnr = calc_psnr(mse)
            print(f"PSNR : {psnr.item()}")

            gt_im = gt_flatten.reshape(hwf[0],hwf[1],3)
            color = color.reshape(hwf[0], hwf[1], 3)
            depth = depth.reshape(hwf[0],hwf[1])
            concat = torch.cat([gt_im,color],1)

            os.makedirs(f"{args.basedir}/img",exist_ok=True)
            os.makedirs(f"{args.basedir}/depth",exist_ok=True)
            os.makedirs(f"{args.basedir}/gt_img",exist_ok=True)
            os.makedirs(f"{args.basedir}/ckpt",exist_ok=True)

            io.imsave(f"{args.basedir}/gt_img/{step:06d}.png",concat.cpu().numpy())
            io.imsave(f"{args.basedir}/img/{step:06d}.png",color.cpu().numpy())
            io.imsave(f"{args.basedir}/depth/{step:06d}.png",depth.cpu().numpy())
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_f_state_dict': optimizer_f.state_dict(),
                'optimizer_p_state_dict': optimizer_p.state_dict(),
                'pose_noise': pose_noise,
                'pose': pose_perturbs
            }, f"{args.basedir}/ckpt/{step:06d}.tar")
