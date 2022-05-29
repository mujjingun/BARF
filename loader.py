import json
import pathlib
from PIL import Image
import numpy as np
from skimage import io
from tqdm import tqdm
import cv2


def load_dataset(dataset_type, basedir, half_res=False, testskip=1):
    """
    Loads dataset.
    :param dataset_type: "llff" or "blender"
    :param basedir:
    :param half_res:
    :param testskip:
    :return: a tuple (images, poses, render_poses, hwf, i_split)
    """
    if dataset_type == "blender":
        return load_blender(basedir, half_res, testskip)
    elif dataset_type == "llff":
        return load_llff(basedir, half_res, testskip)
    else:
        assert False


def load_llff(basedir, half_res, testskip, bound_factor=0.75):
    basedir = pathlib.Path(basedir)
    poses_bounds = np.load(basedir / "poses_bounds.npy")

    # [num_images, 3, 5]
    camera = poses_bounds[:, :-2].reshape([-1, 3, 5])
    poses = camera[..., :4]  # [n, 3, 4]

    # adjust camera poses
    poses[..., 0], poses[..., 1] = poses[..., 1], -poses[..., 0]

    # add 4th row
    poses = np.pad(poses, [(0, 0), (0, 1), (0, 0)])
    poses[:, -1, 3] = 1.0

    # TODO: normalize poses
    # poses = normalize_poses(poses)

    # intrinsics
    height = camera[0, 0, -1]
    width = camera[0, 1, -1]
    focal_length = camera[0, 2, -1]
    hwf = [height, width, focal_length]

    # [num_images, 2]
    bounds = poses_bounds[:, -2:]  # TODO

    img_names = sorted([path for path in (basedir / "images").iterdir()
                        if path.suffix.lower() in ['.jpg', '.png']])

    images = np.stack([io.imread(path) for path in tqdm(img_names, desc="reading images")])
    assert len(images) == poses.shape[0]

    render_poses = np.stack([  # FIXME
        gen_pose(angle, -30., 4.)
        for angle in np.linspace(-180, 180, 41)[:-1]
    ])

    # train-val-test split
    n_train = len(images) // 4 * 3
    n_test = len(images) - n_train
    images = np.concatenate([images[:n_train], images[n_train:], images[n_train:]], axis=0)
    poses = np.concatenate([poses[:n_train], poses[n_train:], poses[n_train:]], axis=0)
    i_split = [
        [i for i in range(n_train)],  # train
        [n_train + i for i in range(n_test)],  # val
        [n_train + i for i in range(n_test)]  # test
    ]

    return images, poses, render_poses, hwf, i_split


def load_blender(basedir, half_res, testskip):
    basedir = pathlib.Path(basedir)

    metadata = {}
    for split in ['train', 'val', 'test']:
        with open(basedir / f'transforms_{split}.json') as fp:
            metadata[split] = json.load(fp)

    images = []
    poses = []
    i_split = []
    idx = 0
    for split in ['train', 'val', 'test']:
        i_split.append([])

        frames = metadata[split]['frames']
        skip = testskip if split == 'test' else 1
        for frame in frames[::skip]:
            img = Image.open(basedir / f"{frame['file_path']}.png")
            images.append(img)
            poses.append(np.array(frame['transform_matrix']))
            rotation = frame['rotation']
            i_split[-1].append(idx)
            idx += 1

        camera_angle_x = metadata[split]['camera_angle_x']
        H, W = images[0].width, images[0].height
        focal_length = .5 * W / np.tan(.5 * camera_angle_x)

    poses = np.stack(poses,0)

    hwf = [H, W, focal_length]

    render_poses = np.stack([
        gen_pose(angle, -30., 4.)
        for angle in np.linspace(-180, 180, 41)[:-1]
    ])

    images = np.stack([np.array(img) for img in images])
    images = images.astype(np.float32) / 255.

    if half_res:
        hwf = [H // 2, W // 2, focal_length / 2.]
        imgs_half_res = np.zeros((images.shape[0], H//2, W//2, 4))
        for i, img in enumerate(images):
            imgs_half_res[i] = cv2.resize(img, (W//2, H//2), interpolation=cv2.INTER_AREA)
        images = imgs_half_res

    # white background
    # images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    return images, poses, render_poses, hwf, i_split


def translate_z(distance):
    M = np.eye(4)
    M[2, 3] = distance
    return M


def rotate_phi(phi):
    M = np.eye(4)
    M[1, 1] = np.cos(phi)
    M[1, 2] = -np.sin(phi)
    M[2, 1] = np.sin(phi)
    M[2, 2] = np.cos(phi)
    return M


def rotate_theta(theta):
    M = np.eye(4)
    M[0, 0] = np.cos(theta)
    M[0, 2] = -np.sin(theta)
    M[2, 0] = np.sin(theta)
    M[2, 2] = np.cos(theta)
    return M


def gen_pose(theta, phi, distance):
    return (
        np.array([
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        @ rotate_theta(np.radians(theta))
        @ rotate_phi(np.radians(phi))
        @ translate_z(distance)
    )


if __name__ == "__main__":
    load_dataset('blender', 'nerf_synthetic/lego')
