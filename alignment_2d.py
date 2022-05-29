import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from skimage import io
import pathlib
from model import PosEncoding
from tqdm import trange, tqdm


def sl3_to_SL3(h):
    # homography: directly expand matrix exponential
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.6151&rep=rep1&type=pdf
    h1, h2, h3, h4, h5, h6, h7, h8 = h.chunk(8, dim=-1)
    A = torch.stack([torch.cat([h5, h3, h1], dim=-1),
                     torch.cat([h4, -h5 - h6, h2], dim=-1),
                     torch.cat([h7, h8, h6], dim=-1)], dim=-2)
    H = A.matrix_exp()
    return H


def gen_grid(h, H, W, M, crop_scale):
    mat = sl3_to_SL3(h)

    grid = F.affine_grid(
        torch.eye(3, device=h.device)[None, :2],
        [1, 3, H, W],
        align_corners=False,
    ) * crop_scale

    grid = torch.cat([grid, torch.ones(1, H, W, 1, device=h.device)], -1)  # [1, H, W, 3]
    grid = torch.einsum('mdt,mhwd->mhwt', mat, grid.expand(M, -1, -1, -1))
    grid = (grid / grid[..., -1:])[..., :-1]
    return grid


# pick random patches
def generate_patches(image, num_patches, noise_scale, crop_scale):
    H, W = image.shape[1:]
    M = num_patches
    image = image[None].expand(M, -1, -1, -1)

    h = torch.randn(M, 8, device=image.device) * noise_scale
    h[0] = 0.0  # identity for the first patch

    grid = gen_grid(h, H, W, num_patches, crop_scale)
    patches = F.grid_sample(image, grid, align_corners=False)
    return patches, h


# visualize patches and save them to image files
def visualize_patches(patches, num_patches, basedir):
    plt.figure(figsize=(num_patches, 1), dpi=300)
    for i in range(num_patches):
        plt.subplot(1, num_patches, i + 1)
        plt.imshow(patches[i].permute(1, 2, 0).cpu())
        plt.axis('off')
    plt.savefig(basedir / "patches.png", bbox_inches='tight')


def visualize_corners(image, true_poses, crop_scale, num_patches, basedir, filename="warps.png"):
    H, W = image.shape[1], image.shape[2]
    mat = sl3_to_SL3(true_poses)
    corners = torch.tensor([
        [-crop_scale, -crop_scale, 1.0],
        [-crop_scale, +crop_scale, 1.0],
        [+crop_scale, +crop_scale, 1.0],
        [+crop_scale, -crop_scale, 1.0],
        [-crop_scale, -crop_scale, 1.0],
    ], device=image.device)
    corners = torch.einsum('ndt,kd->nkt', mat, corners)
    corners = (corners / corners[:, :, -1:])[..., :-1]
    corners[..., 0] = (corners[..., 0] + 1.0) * 0.5 * W
    corners[..., 1] = (corners[..., 1] + 1.0) * 0.5 * H

    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(image.permute(1, 2, 0).cpu())
    for i in range(num_patches):
        plt.plot(corners[i, :, 0].cpu(), corners[i, :, 1].cpu(), color=f'C{i}', label=f"Patch {i}")
    plt.legend()
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.savefig(basedir / filename, bbox_inches='tight')


class Model(torch.nn.Module):
    def __init__(self, pos_enc):
        super(Model, self).__init__()
        self.pos_enc = pos_enc

        self.features = torch.nn.ModuleList([
            torch.nn.Linear(pos_enc.encode_dim, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 3),
        ])

    def forward(self, x, step):
        # x: [batch_size, 2]
        # output: color [batch_size, 3]
        x = self.pos_enc.encode(x, step)
        for feat in self.features[:-1]:
            x = feat(x)
            x = F.softplus(x)
        x = self.features[-1](x)
        x = torch.sigmoid(x)
        return x


@torch.no_grad()
def visualize_f(model, device, H, W, step, basedir, filename='image.png'):
    grid = gen_grid(torch.zeros(1, 8, device=device), H, W, 1, 1.0)
    grid = grid.reshape(-1, 2)
    pixels = model(grid, step)
    pixels = pixels.reshape(H, W, 3)

    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(pixels.cpu())
    plt.savefig(basedir / filename, bbox_inches='tight')


def train(args, image, patches, true_poses, basedir):
    # patches: (M, 3, H, W)
    # true_poses: (M, 8)
    poses = torch.nn.Parameter(torch.zeros(args.num_patches - 1, 8, device=image.device))
    model = Model(PosEncoding(2, 8, 0, 2000)).to(image.device)
    H, W = patches.shape[2:]
    M = len(patches)

    if (basedir / 'ckpt_4999.pth').exists():
        ckpt = torch.load(basedir / 'ckpt_4999.pth', map_location=image.device)
        model.load_state_dict(ckpt['state_dict'])
        poses.data.copy_(ckpt['poses'])
        visualize_f(model, image.device, H, W, 4999, basedir)
        return

    opt = torch.optim.Adam(
        [*model.parameters(), poses],
        lr=0.001
    )

    for step in trange(5000):
        opt.zero_grad()

        poses_fixed = torch.cat([torch.zeros(1, 8, device=poses.device), poses], 0)

        grid = gen_grid(poses_fixed, H, W, args.num_patches, args.crop_scale)  # (M, H, W, 2)
        grid = grid.reshape(-1, 2)  # (MHW, 2)

        prediction = model(grid, step)  # (MHW, 3)
        prediction = prediction.reshape(M, H, W, 3)  # (M, H, W, 3)
        ground_truth = patches.permute(0, 2, 3, 1)  # (M, H, W, 3)

        loss = ((prediction - ground_truth)**2).sum()
        tqdm.write(f"loss={loss:.4f}")

        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            visualize_corners(image, poses_fixed.data, args.crop_scale, M, basedir, f"warps_{step}.png")
            visualize_f(model, image.device, H, W, step, basedir, f"image_{step}.png")

        if (step + 1) % 100 == 0:
            torch.save({
                'true_poses': true_poses,
                'poses': poses,
                'state_dict': model.state_dict()
            }, basedir / f"ckpt_{step}.pth")

    return poses, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="../data/cat.jpg")
    parser.add_argument('--num_patches', type=int, default=5)
    parser.add_argument('--basedir', type=str, default='planar/')
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--crop_scale', type=float, default=0.4)
    args = parser.parse_args()

    basedir = pathlib.Path(args.basedir)
    basedir.mkdir(parents=True, exist_ok=True)

    image = torch.tensor(io.imread(args.image_path)).float() / 255.
    image = image.permute(2, 0, 1)
    image = image.cuda()

    patches, true_poses = generate_patches(image, args.num_patches, args.noise_scale, args.crop_scale)

    visualize_patches(patches, args.num_patches, basedir)
    visualize_corners(image, true_poses, args.crop_scale, args.num_patches, basedir)

    train(args, image, patches, true_poses, basedir)


if __name__ == "__main__":
    main()
