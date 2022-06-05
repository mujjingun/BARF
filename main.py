import argparse
import loader
from model import *
from train2 import train_nerf
from test import test_nerf
from load_llff import *
import torch
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def main():
    parser = argparse.ArgumentParser('BARF')
    parser.add_argument('--dataset_type', type=str, default='blender')
    parser.add_argument('--datadir', type=str, default='../data/nerf_synthetic/lego')
    parser.add_argument('--half_res', default=False, action='store_true')
    parser.add_argument('--testskip', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=200000)
    parser.add_argument('--num_rays', type=int, default=1024)
    parser.add_argument('--num_points', type=int, default=128)
    parser.add_argument('--pos_enc_L', type=int, default=10)
    parser.add_argument('--dir_enc_L', type=int, default=4)
    parser.add_argument('--basedir', type=str, default='test_result')
    parser.add_argument('--add_perturb', default=False, action='store_true')
    parser.add_argument('-c2f', '--coarse_to_fine', default=False, action='store_true')
    parser.add_argument('--c2f_begin', type=int, default=20000)
    parser.add_argument('--c2f_end', type=int, default=100000)
    parser.add_argument('--without_pos_enc', default=False, action='store_true')
    parser.add_argument('--lr_f_start', type=float, default=5e-4)
    parser.add_argument('--lr_f_end', type=float, default=1e-4)
    parser.add_argument('--lr_p_start', type=float, default=1e-3)
    parser.add_argument('--lr_p_end', type=float, default=1e-5)
    parser.add_argument('--white_background', default=False, action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--auto_load', default=False, action='store_true')
    parser.add_argument('--reverse_order', default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(f"{args.basedir}",exist_ok=True)

    images, poses, render_poses, hwf, i_split, bounds = loader.load_dataset(
        args.dataset_type, args.datadir, args.half_res, args.testskip
    )

    near = 2. if args.dataset_type == 'blender' else np.ndarray.min(bounds)*0.9
    far = 6. if args.dataset_type == 'blender' else np.ndarray.max(bounds)*1.

    if args.dataset_type == 'llff':
        llff_holdout = 8
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, llff_holdout,
                                                                    recenter=True, bd_factor=.75,
                                                                    spherify=False)

        print(poses.shape)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        print('Auto LLFF holdout,', llff_holdout)
        i_test = np.arange(images.shape[0])[::llff_holdout]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.

        i_split = [i_train, i_val, i_test]
        print(i_train)

    print(images.shape)
    print(poses.shape)
    print(near,far)

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.white_background:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    if images.shape[-1] == 4:
        images = images[...,:3] * images[..., -1:]


    pos_encoding = PosEncoding(3,L=args.pos_enc_L,lower_bound=args.c2f_begin,upper_bound=args.c2f_end)
    dir_encoding = PosEncoding(3,L=args.dir_enc_L,lower_bound=args.c2f_begin,upper_bound=args.c2f_end)

    in_dim = pos_encoding.ret_encode_dim()
    in_view_dim = dir_encoding.ret_encode_dim()

    pos_encoder = lambda x, step ,encoder=pos_encoding : encoder.encode(x,step)
    dir_encoder = lambda x, step ,encoder=dir_encoding : encoder.encode(x,step)

    if args.without_pos_enc:
        in_dim = 3
        in_view_dim = 3
        pos_encoder = None
        dir_encoder = None

    model = NeRFModel(in_dim=in_dim, in_view_dim=in_view_dim).to(device)

    # pos_encoder.to(device)
    # dir_encoder.to(device)
    optimizer_state = None
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_f_state = checkpoint['optimizer_f_state_dict']
        optimizer_p_state = checkpoint['optimizer_p_state_dict']
        pose_noise = checkpoint['pose_noise']
        pose_perturbs = checkpoint['pose']
    
    if args.auto_load:
        checkpoint = torch.load(os.path.join(args.basedir,"ckpt","200000.tar"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_f_state = checkpoint['optimizer_f_state_dict']
        optimizer_p_state = checkpoint['optimizer_p_state_dict']
        pose_noise = checkpoint['pose_noise']
        pose_perturbs = checkpoint['pose']


    if args.test:
        test_nerf(
            model, pos_encoder, dir_encoder,
            images, poses, render_poses, hwf, i_split, device, near, far, bounds,
            pose_noise, pose_perturbs,
            args
        )

    else:
        pose_params = train_nerf(
            model, pos_encoder, dir_encoder,
            images, poses, render_poses, hwf, i_split, device, near, far, bounds,
            args
        )


if __name__ == "__main__":
    main()
