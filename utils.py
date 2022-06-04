import torch
import numpy as np
import pytorch3d.transforms
import torch.nn.functional as F
import math


def procrustes_analysis(X0, X1):  # [N, 3]
    # translation
    mu0 = X0.mean(dim=0, keepdim=True)
    mu1 = X1.mean(dim=0, keepdim=True)
    X0 = X0 - mu0
    X1 = X1 - mu1

    # scale
    s0 = (X0 ** 2).sum(dim=-1).mean().sqrt()  # TODO replace with std
    s1 = (X1 ** 2).sum(dim=-1).mean().sqrt()
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


@torch.no_grad()
def pose_distance(truth, perturbs):
    # obtain camera positions
    # truth, perturbs = invert(truth), invert(perturbs)  # (N, 3, 4)

    origin = torch.tensor([[[0., 0., 0., 1.]]], device=truth.device)
    truth_origin = (origin @ invert(truth).transpose(-1, -2))[:,0]
    perturbs_origin = (origin @ invert(perturbs).transpose(-1, -2))[:,0]  # (N, 3)

    # perform procrustes analysis
    try:
        truth_mu, perturbs_mu, truth_scale, perturbs_scale, rotation = procrustes_analysis(truth_origin, perturbs_origin)
    except np.linalg.LinAlgError:
        print("SVD did not converge")
        return

    # align the perturbed origin to ground truth
    aligned_origin = ((perturbs_origin - perturbs_mu) / perturbs_scale) @ rotation.T * truth_scale + truth_mu  # (N, 3)
    R_aligned = perturbs[..., :3] @ rotation.T  # (N, 3, 3)
    t_aligned = (-R_aligned @ aligned_origin[...,None])[...,0]
    aligned_mat = to_matrix(R_aligned, t_aligned)

    trans_dist = (truth_origin - aligned_origin).norm(dim=-1).mean()

    direc = torch.tensor([[[0., 0., 1., 0.]]], device=truth.device)
    truth_direc = (direc @ invert(truth).transpose(-1, -2))[:, 0]
    aligned_direc = (direc @ invert(aligned_mat).transpose(-1, -2))[:, 0]  # (N, 3)

    angle_dist = torch.rad2deg(torch.acos(F.cosine_similarity(truth_direc.double(), aligned_direc.double())))
    angle_dist = angle_dist.mean(0)

    print(f"pose error: trans = {trans_dist.item():.4f} / "
          f"angle = {angle_dist.item():.4f}")
