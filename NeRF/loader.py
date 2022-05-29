import json
import pathlib
from PIL import Image
import numpy as np
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
    assert dataset_type == "blender"

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
