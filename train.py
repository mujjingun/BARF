import math
import torch
from sample_points import *
from estimate_color import estimate_color
import numpy as np
from tqdm import tqdm, trange
import os
from skimage import io

mse_loss = lambda output, gt: torch.mean((output - gt) ** 2)


class Lie:
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self, w):  # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def SO3_to_so3(self, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
                    ..., None, None] % np.pi  # ln(R) will explode if theta==pi
        lnR = 1 / (2 * self.taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    def se3_to_SE3(self, wu):  # [...,3]
        w, u = wu.split([3, 3], dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I + A * wx + B * wx @ wx
        V = I + B * wx + C * wx @ wx
        Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
        return Rt

    def SE3_to_se3(self, Rt, eps=1e-8):  # [...,3,4]
        R, t = Rt.split([3, 1], dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta ** 2 + eps) * wx @ wx
        u = (invV @ t)[..., 0]
        wu = torch.cat([w, u], dim=-1)
        return wu

    def skew_symmetric(self, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                          torch.stack([w2, O, -w0], dim=-1),
                          torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    def taylor_A(self, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0: denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_B(self, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    def taylor_C(self, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans


def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1 @ R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()  # numerical stability near -1/+1
    return angle


def pose_distance(perturbs):
    trans_dist = torch.sqrt((perturbs[:, :3] ** 2).sum(1)).mean(0)
    angle_dist = rotation_distance(
        Lie().se3_to_SE3(perturbs)[:, :, :3],
        torch.eye(3, device=perturbs.device)[None].expand(perturbs.shape[0], 3, 3)
    ).mean(0)

    print(f"pose RMSE: trans = {trans_dist.item():.4f} / "
          f"angle = {math.degrees(angle_dist.item()):.4f}")


def to_matrix(pose_params):
    rotation_mat = Lie().se3_to_SE3(pose_params)  # eulerAnglesToRotationMatrix(angles)
    c2w = rotation_mat.new_zeros(pose_params.shape[0], 4, 4)
    c2w[:, :3, :] = rotation_mat
    c2w[:, 3, 3] = 1.0
    return c2w.float()


def train_nerf(model, pos_encoder, dir_encoder,
               images, poses, render_poses, hwf, i_split, device,
               args):
    poses = torch.tensor(poses, dtype=torch.float32, device=device)

    # add perturbations on the camera pose
    pose_perturbs = torch.nn.Parameter(
        torch.cat([
            torch.randn((poses.shape[0], 6), device=device) * 0.15
        ], dim=1)
    )
    pose_distance(pose_perturbs)

    lr_f_start, lr_f_end = 5e-4, 1e-4
    lr_f_gamma = (lr_f_end / lr_f_start) ** (1. / args.n_steps)
    lr_p_start, lr_p_end = 1e-3, 1e-5
    lr_p_gamma = (lr_p_end / lr_p_start) ** (1. / args.n_steps)

    optimizer = torch.optim.Adam(
        params=[
            {'params': model.parameters(), 'lr': lr_f_start},
            {'params': [pose_perturbs], 'lr': lr_p_start},
        ],
        betas=(0.9, 0.999),
    )

    loss_running_avg = None

    pbar = tqdm(range(args.n_steps))
    for step in pbar:
        optimizer.zero_grad()

        # sample points on ray
        i_train = i_split[0]
        train_idx = np.random.choice(i_train, 1)

        train_im = images[train_idx]  # H x W x 3

        c2w = to_matrix(pose_perturbs[train_idx]) @ poses[train_idx]
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

        color = estimate_color(model, sampled_points, sampled_directions, lin, pos_encoder, dir_encoder,
                               step if args.coarse_to_fine else -1)

        # compute loss
        loss = mse_loss(color, gt)
        if loss_running_avg is None:
            loss_running_avg = loss
        else:
            loss_running_avg = 0.99 * loss_running_avg + 0.01 * loss
        pbar.set_description(f"loss : {loss:06f} | {loss * 1000:02f} ({loss_running_avg * 1000:02f}) "
                             f"lr_f={optimizer.param_groups[0]['lr']} lr_p={optimizer.param_groups[1]['lr']}")

        loss.backward()
        pose_grad = pose_perturbs.grad[train_idx].clone()
        optimizer.step()

        optimizer.param_groups[0]['lr'] *= lr_f_gamma
        optimizer.param_groups[1]['lr'] *= lr_p_gamma

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

            color = color.reshape(hwf[0], hwf[1], 3)

            os.makedirs(f"{args.basedir}/img", exist_ok=True)
            os.makedirs(f"{args.basedir}/ckpt", exist_ok=True)
            io.imsave(f"{args.basedir}/img/{step}.png", color.cpu().numpy())
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pose': pose_perturbs
            }, f"{args.basedir}/ckpt/{step}.tar")
