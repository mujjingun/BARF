import torch
import numpy as np
import os
from sample_points import *
from estimate_color import estimate_color
from tqdm import tqdm, trange
import imageio as io
import math
from skimage.metrics import structural_similarity as calc_ssim
from lpips import LPIPS

mse_loss = lambda output, gt : torch.mean((output-gt)**2)
calc_psnr = lambda mse : -10*torch.log10(mse)

def test_nerf(model, pos_encoder, dir_encoder,
              images, poses, render_poses, hwf, i_split, device, near, far,
              args):

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    psnr_ = []
    ssim_ = []
    lpips_ = []
    calc_lpips = LPIPS(net='alex').to(device)
    with torch.no_grad():
        i_test = i_split[2]
        for idx in i_test:
            test_im = images[idx]
            gt_flatten = torch.from_numpy(test_im.reshape(-1,3)).to(device)

            c2w = torch.from_numpy(poses[idx]).to(device)
            c2w = c2w.type(torch.float32)

            world_o, world_d = get_rays(hwf,c2w)
            world_d_flatten = world_d.reshape(-1,3)

            selected_d = world_d_flatten

            sampled_points, sampled_directions, lin = sample_points(
                world_o, selected_d, args.num_points, device, near, far
            )

            colors = []
            depths = []
            total_pixel = hwf[0]*hwf[1]
            batch_size = 8000
            iter = total_pixel//batch_size

            for j in trange(iter):
                batch_points = sampled_points[batch_size*j:batch_size*(j+1)]
                batch_directions = sampled_directions[batch_size*j:batch_size*(j+1)]
                batch_lin = lin[batch_size*j:batch_size*(j+1)]
                color, depth = estimate_color(model, batch_points, batch_directions, batch_lin, pos_encoder, dir_encoder, args.white_background)
                colors.append(color)
                depths.append(depth)

            color = torch.cat(colors,0)
            depth = torch.cat(depths,0)

            mse = mse_loss(color,gt_flatten)
            psnr = calc_psnr(mse)
            psnr_.append(psnr.cpu().numpy())

            gt_im = gt_flatten.reshape(hwf[0],hwf[1],3)
            
            color = color.reshape(hwf[0],hwf[1],3)

            ssim = calc_ssim(color.clone().cpu().numpy(), gt_im.clone().cpu().numpy(), channel_axis = 2, multichannel=True)
            ssim_.append(ssim)

            gt_im = gt_im.type('torch.cuda.FloatTensor')
            lpips = calc_lpips(color.transpose(1,2).transpose(0,1).unsqueeze(0), gt_im.transpose(1,2).transpose(0,1).unsqueeze(0))
            lpips_.append(lpips.cpu().numpy())

            print(f"[Image index {idx}] PSNR : {psnr.item()} | SSIM : {ssim} | LPIPS : {lpips.item()}")
            depth = depth.reshape(hwf[0],hwf[1])

            concat = torch.cat([gt_im,color],1)
            os.makedirs(f"{args.basedir}/test_img",exist_ok=True)
            os.makedirs(f"{args.basedir}/test_depth",exist_ok=True)
            os.makedirs(f"{args.basedir}/test_gt_img",exist_ok=True)

            io.imsave(f"{args.basedir}/test_gt_img/{idx:03d}.png",concat.cpu().numpy())
            io.imsave(f"{args.basedir}/test_img/{idx:03d}.png",color.cpu().numpy())
            io.imsave(f"{args.basedir}/test_depth/{idx:03d}.png",depth.cpu().numpy())
        
        psnr_ = np.mean(np.stack(psnr_,0))
        ssim_ = np.mean(np.stack(ssim_,0))
        lpips_ = np.mean(np.stack(lpips_,0))

        print(f"psnr : {psnr_} | ssim : {ssim_} | lpips : {lpips_}")

        metrics = np.array([psnr_,ssim_,lpips_])
        np.save(f"{args.basedir}/metric.npy",metrics)