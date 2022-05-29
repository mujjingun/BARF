import argparse
import loader
from model import *
from train import train_nerf
import torch
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

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

    args = parser.parse_args()

    os.makedirs(f"{args.basedir}",exist_ok=True)

    images, poses, render_poses, hwf, i_split = loader.load_dataset(
        args.dataset_type, args.datadir, args.half_res, args.testskip
    )


    white_bkgd = False
    if white_bkgd:
        # white background
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])


    if images.shape[-1] == 4:
        images = images[...,:3]

    

    pos_encoding = PosEncoding(3,L=args.pos_enc_L,upper_bound=args.n_steps/20)
    dir_encoding = PosEncoding(3,L=args.dir_enc_L,upper_bound=args.n_steps/20)

    pos_encoder = lambda x, step ,encoder=pos_encoding : encoder.encode(x,step)
    dir_encoder = lambda x, step ,encoder=dir_encoding : encoder.encode(x,step)
    
    in_dim = pos_encoding.ret_encode_dim()
    in_view_dim = dir_encoding.ret_encode_dim()

    model = NeRFModel2(in_dim=in_dim, in_view_dim=in_view_dim).to(device)


    # pos_encoder.to(device)
    # dir_encoder.to(device)


    train_nerf(
        model, pos_encoder, dir_encoder,
        images, poses, render_poses, hwf, i_split, device,
        args
    )

    # TODO: evaluate trained model


# torch.set_default_tensor_type('torch.cuda.FloatTensor')
main()