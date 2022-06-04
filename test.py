import torch
import numpy as np
import os
from sample_points import *
from estimate_color import estimate_color
from tqdm import tqdm, trange
import imageio as io
import math
import pytorch3d.transforms
from utils import *
from skimage.metrics import structural_similarity as calc_ssim
from lpips import LPIPS

mse_loss = lambda output, gt: torch.mean((output - gt) ** 2)
calc_psnr = lambda mse: -10 * torch.log10(mse)


def get_align(truth, perturbs):

    # obtain camera positions
    truth, perturbs = invert(truth), invert(perturbs)  # (N, 3, 4)

    origin = torch.tensor([[0., 0., 0., 1.]], device=truth.device).unsqueeze(-1)
    truth_origin = (truth @ origin).squeeze(2)
    perturbs_origin = (perturbs @ origin).squeeze(2)  # (N, 3)

    # perform procrustes analysis
    return procrustes_analysis(truth_origin, perturbs_origin)


@torch.no_grad()
def test_nerf(model, pos_encoder, dir_encoder,
              images, poses, render_poses, hwf, i_split, device, near, far, bounds,
              pose_noise, pose_perturbs,
              args):
    i_train, i_val, i_test = i_split

    train_poses = torch.tensor(poses, dtype=torch.float32, device=device)[i_train]
    test_poses = torch.tensor(poses, dtype=torch.float32, device=device)[i_test]

    train_poses_inv = invert(train_poses[:, :3, :])
    calc_poses = (train_poses_inv @
                  pytorch3d.transforms.se3_exp_map(pose_noise).mT @
                  pytorch3d.transforms.se3_exp_map(pose_perturbs).mT)
    truth_mu, perturbs_mu, truth_scale, perturbs_scale, rotation = get_align(train_poses_inv, calc_poses)
    pose_distance(train_poses_inv, calc_poses)

    origin = torch.tensor([[0., 0., 0., 1.]], device=device).unsqueeze(-1)
    test_origin = (test_poses[:, :3, :] @ origin).squeeze(2)
    aligned_origin = ((test_origin - truth_mu) / truth_scale) @ rotation * perturbs_scale + perturbs_mu  # (N, 3)

    test_poses_inv = invert(test_poses[:, :3, :])
    R_aligned = test_poses_inv[..., :3] @ rotation
    t_aligned = (-R_aligned @ aligned_origin[..., None])[..., 0]
    test_poses_inv = to_matrix(R_aligned, t_aligned)
    test_poses = expand(invert(test_poses_inv))

    psnr_ = []
    ssim_ = []
    lpips_ = []
    calc_lpips = LPIPS(net='alex').to(device)
    for idx in i_test:
        test_im = images[idx]
        gt_flatten = torch.from_numpy(test_im.reshape(-1, 3)).to(device)

        c2w = test_poses[idx - i_test[0]]
        c2w = c2w.type(torch.float32)

        world_o, world_d = get_rays(hwf, c2w)
        world_d_flatten = world_d.reshape(-1, 3)

        selected_d = world_d_flatten

        colors = []
        depths = []
        total_pixel = hwf[0] * hwf[1]
        batch_size = 8000 // max(1, (args.num_points // 256))
        iter = total_pixel // batch_size

        for j in trange(iter):
            batch_points, batch_directions, batch_lin = sample_points(
                world_o, selected_d[batch_size * j:batch_size * (j + 1)], args.num_points, device
            )
            color, depth = estimate_color(model, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder,
                                          -1, args.white_background)
            colors.append(color)
            depths.append(depth)

        color = torch.cat(colors, 0)
        depth = torch.cat(depths, 0)

        mse = mse_loss(color, gt_flatten)
        psnr = calc_psnr(mse)
        psnr_.append(psnr.cpu().numpy())

        print(f"[Image index {idx}] PSNR : {psnr.item()}")

        gt_im = gt_flatten.reshape(hwf[0], hwf[1], 3)

        color = color.reshape(hwf[0], hwf[1], 3)

        ssim = calc_ssim(color.clone().cpu().numpy(), gt_im.clone().cpu().numpy(), channel_axis = 2, multichannel=True)
        ssim_.append(ssim)

        gt_im = gt_im.type('torch.cuda.FloatTensor')
        lpips = calc_lpips(color.transpose(1,2).transpose(0,1).unsqueeze(0), gt_im.transpose(1,2).transpose(0,1).unsqueeze(0))
        lpips_.append(lpips.cpu().numpy())

        print(f"[Image index {idx}] PSNR : {psnr.item()} | SSIM : {ssim} | LPIPS : {lpips.item()}")
        
        
        depth = depth.reshape(hwf[0], hwf[1])

        concat = torch.cat([gt_im, color], 1)
        os.makedirs(f"{args.basedir}/test_img", exist_ok=True)
        os.makedirs(f"{args.basedir}/test_depth", exist_ok=True)
        os.makedirs(f"{args.basedir}/test_gt_img", exist_ok=True)

        io.imsave(f"{args.basedir}/test_gt_img/{idx:03d}.png", concat.cpu().numpy())
        io.imsave(f"{args.basedir}/test_img/{idx:03d}.png", color.cpu().numpy())
        io.imsave(f"{args.basedir}/test_depth/{idx:03d}.png", depth.cpu().numpy())
    psnr_ = np.mean(np.stack(psnr_,0))
    ssim_ = np.mean(np.stack(ssim_,0))
    lpips_ = np.mean(np.stack(lpips_,0))

    metrics = np.array([psnr_,ssim_,lpips_])
    np.save(f"{args.basedir}/metric.npy",metrics)