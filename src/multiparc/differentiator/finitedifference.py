import torch
import torch.nn as nn
import torch.nn.functional as F
from multiparc.boundary_conditions import PaddingAll


class FiniteDifferenceGrad(nn.Module):
    def __init__(
        self,
        filter_1d=torch.tensor([-1.0, 1.0], dtype=torch.float32),
        device="cuda",
        right_bottom=True,
        custom_padding=PaddingAll("reflect", 0),
    ):
        """
        Module for calculation of gradient with finite difference filter.
    
        Args:
            filter_1d: torch.tensor, optional, default ```[-1.0, 1.0]```. 1D finite difference filter for gradient calculation. Default value is one-sided only
            device: str, optional, default ```cuda```. The device to store the filters in.
            right_bottom: bool, optional, default ```True```. Whether to pad more to the right/bottom (True) or left/top (False). No effect if the padding is symmetric (odd-sized filters).
            custom_padding: nn.Module, optional, default ```PaddingAll("reflect", 0)````. Custom padding module for boundary condition enforcement. Default value corresponds to zero gradient on all boundaries.
        """
        super().__init__()
        self.padding = custom_padding
        # Determine padding for dy and dx based on filter size
        filter_size = filter_1d.shape[0]
        if filter_size % 2 == 1:
            pad_top = pad_bottom = (filter_size - 1) // 2
            pad_left = pad_right = (filter_size - 1) // 2
        elif right_bottom:
            pad_top = (filter_size - 1) // 2
            pad_bottom = pad_top + 1
            pad_left = (filter_size - 1) // 2
            pad_right = pad_left + 1
        else:
            pad_bottom = (filter_size - 1) // 2
            pad_top = pad_bottom + 1
            pad_right = (filter_size - 1) // 2
            pad_left = pad_right + 1
        self.dy_pad = [0, 0, pad_top, pad_bottom]  # (left, right, top, bottom)
        self.dx_pad = [pad_left, pad_right, 0, 0]  # (left, right, top, bottom)
        # Initialize dy_conv weights
        dy_filter = filter_1d.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        dy_filter.requires_grad = False
        self.register_buffer(
            "dy_filter", dy_filter.to(device)
        )  # [1, 1, filter_size, 1]
        # Initialize dx_conv weights
        dx_filter = filter_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        dx_filter.requires_grad = False
        self.register_buffer(
            "dx_filter", dx_filter.to(device)
        )  # [1, 1, 1, filter_size]

    def forward(self, x):
        """ 
        Forward pass.

        Args:
            x: torch.tensor of shape (batch_size, n_channel, y, x).

        Returns:
            dy: torch.tensor with the same shape as ```x```. Derivative along y-axis.
            dx: torch.tensor with the same shape as ```x```. Derivative along x-axis.
        """
        # Compute dy: vertical derivative
        x_padded_dy = self.padding(x, self.dy_pad)
        dy = F.conv2d(
            x_padded_dy,
            self.dy_filter.expand(x.shape[1], -1, -1, -1),
            groups=x.shape[1],
        )
        # Compute dx: horizontal derivative
        x_padded_dx = self.padding(x, self.dx_pad)
        dx = F.conv2d(
            x_padded_dx,
            self.dx_filter.expand(x.shape[1], -1, -1, -1),
            groups=x.shape[1],
        )
        return dy, dx
