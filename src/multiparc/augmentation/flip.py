import torch


def flip_image(x, target, flip_x=True, flip_y=True, flip_prob=0.5):
    """
    x, target: (batch_size, channels, y, x). Last 2 channels must be (x_velocity, y_velocity).
    """
    if flip_x and torch.rand(1).item() < flip_prob:
        x = x.flip(-1)
        x[:, -2] = -x[:, -2]
        target = target.flip(-1)
        target[:, :, -2] = -target[:, :, -2]
    if flip_y and torch.rand(1).item() < flip_prob:
        x = x.flip(-2)
        x[:, -1] = -x[:, -1]
        target = target.flip(-2)
        target[:, :, -1] = -target[:, :, -1]
    return x, target
