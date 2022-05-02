import argparse
import loader
from model import NeRFModel
from train import train_nerf


def main():
    parser = argparse.ArgumentParser('BARF')
    parser.add_argument('--dataset_type', type=str, default='blender')
    parser.add_argument('--basedir', type=str, default='nerf_synthetic/lego')
    parser.add_argument('--half_res', default=False, action='store_true')
    parser.add_argument('--testskip', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_rays', type=int, default=1024)
    parser.add_argument('--num_points', type=int, default=256)

    args = parser.parse_args()

    images, poses, render_poses, hwf, i_split = loader.load_dataset(
        args.dataset_type, args.basedir, args.half_res, args.testskip
    )

    model = NeRFModel()

    train_nerf(
        model,
        images, poses, render_poses, hwf, i_split,
        args
    )

    # TODO: evaluate trained model

