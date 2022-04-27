import torch
from sample_points import sample_points
from estimate_color import estimate_color


def train_nerf(model,
               images, poses, render_poses, hwf, i_split,
               args):

    optimizer = torch.optim.Adam(
        model.parameters(),
        5e-4, args.weight_decay
    )

    for step in range(args.n_steps):
        optimizer.zero_grad()

        # TODO: sample points on ray
        sampled_points, sampled_directions = sample_points(
            # TODO
        )

        color = estimate_color(model, sampled_points, sampled_directions)

        # TODO: compute loss
        loss = nerf_loss(color, images)

        loss.backward()
        optimizer.step()
