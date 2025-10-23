import torch
import torch.nn as nn
import torch.nn.functional as F
from multiparc.boundary_conditions import PaddingAll


class Diffusion(nn.Module):
    """
    Computes the Laplacian of the state variable using finite difference filters.
    """

    def __init__(
        self,
        filter_2d=torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
        device="cuda",
        custom_padding=PaddingAll("reflect", 0),
    ):
        '''
        Constructor of differentiator.diffusion.Diffusion

        Args:
            filter_2d: 2d torch.tensor, optional, default ```[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]```. Finite difference filter that will be used to calculate advection.
            device: str, optional, default ```cuda```. Device where the tensors and modules will be stored.
            custom_padding: nn.Module, optional, default ```PaddingAll("reflect", 0)```. Custom padding module for boundary condition enforcement. Default values applies zero gradient on all boundaries.
        '''
        super().__init__()
        self.padding = custom_padding
        # Determine padding for dy and dx based on filter size
        filter_size = filter_2d.shape
        assert len(filter_size) == 2
        assert filter_size[0] == filter_size[1]
        assert filter_size[0] % 2 == 1
        self.padding_instructions = [
            (filter_size[0] - 1) // 2,
        ] * 4  # (left, right, top, bottom)
        # Initialize conv weights
        dx2dy2_filter = filter_2d.unsqueeze(0).unsqueeze(0)
        dx2dy2_filter.requires_grad = False
        self.register_buffer(
            "dx2dy2_filter", dx2dy2_filter.to(device)
        )  # [C, 1, filter_size, filter_size]

    def forward(self, x):
        """
        Forward pass function.

        Args:
            x: torch.tensor of shape (batch_size, n_channels, y, x).

        Returns:
            lap: torch.tensor of the same shape with ```x```. Computed Laplacian of the tensor.
        """
        x_padded = self.padding(x, self.padding_instructions)
        lap = F.conv2d(
            x_padded,
            self.dx2dy2_filter.expand(x.shape[1], -1, -1, -1),
            groups=x.shape[1],
        )
        return lap
