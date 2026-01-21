import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE) layer implementation in PyTorch.

    This class normalizes the input feature map and modulates it with scaling (gamma) and shifting (beta)
    parameters that are functions of a spatially-varying mask.

    Args:
        in_channels (int): Number of channels in the input feature map.
        mask_channels (int): Number of channels in the input mask.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        activation (nn.Module, optional): Activation function. Default is nn.ReLU.
        activation_args (dict, optional): Args to pass to activation function. Default is an empty dictionary.
        eps (float, optional): Small constant for numerical stability. Default is 1e-5.
        custom_padding (nn.Module, optional): User customized padding for boundary condition enforcement.
    """

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        kernel_size: int,
        activation: nn.Module,
        activation_args: dict,
        eps: float,
        custom_padding: nn.Module,
    ):
        super(SPADE, self).__init__()
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Feature normalization
        self.feature_norm = nn.InstanceNorm2d(
            in_channels, eps=eps, affine=False, track_running_stats=False
        )
        # Mask conv
        self.mask_conv = nn.Sequential(
            nn.Conv2d(
                mask_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=0,
            ),
            activation(**activation_args),
        )
        # Convolutional layers to generate gamma and beta parameters
        self.gamma_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=0,
        )
        self.beta_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=0,
        )

    def forward(self, x, mask):
        """
        Forward pass of the SPADE layer.

        Args:
            x (torch.Tensor): Input feature map to be normalized. Shape: [N, C, H, W].
            mask (torch.Tensor): Input mask providing spatial modulation. Shape: [N, M, H, W].

        Returns:
            torch.Tensor: The output tensor after applying SPADE normalization. Shape: [N, C, H, W].
        """
        # Mask to gamma and beta
        mask_padded = self.padding(mask, self.padding_instruction)
        mask_feat = self.mask_conv(mask_padded)
        mask_feat_padded = self.padding(mask_feat, self.padding_instruction)
        gamma = self.gamma_conv(mask_feat_padded)  # Scale parameter
        beta = self.beta_conv(mask_feat_padded)  # Shift parameter
        # Normalize input feature map
        x_normalized = self.feature_norm(x)
        # Rescale the normalized feature map
        out = gamma * x_normalized + beta
        return out


class SPADEGeneratorUnit(nn.Module):
    """
    SPADE Generator Unit implementation in PyTorch.

    This module represents a SPADE block used in generator architectures, consisting of:
    - Gaussian noise addition
    - Two sequential SPADE-Conv blocks with LeakyReLU activations
    - A skip connection with a SPADE-Conv block

    Args:
        in_channels (int): Number of channels in the input feature map `x`.
        out_channels (int): Number of output channels after convolution.
        mask_channels (int): Number of channels in the input mask `mask`.
        kernel_size (int, optional): Size of the convolutional kernels not in SPADE. Default is 1.
        spade_kernel_size (int, optional): Size of the convolutional kernels in SPADE. Default is 3.
        padding_mode (str, optional): Padding mode for `F.pad`. Default is 'constant'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask_channels: int,
        kernel_size: int = 1,
        activation: nn.Module = nn.LeakyReLU,
        activation_args: dict = {"negative_slope": 0.2},
        noise_std: float = 0.05,
        skip_spade: bool = False,
        spade_kernel_size: int = 3,
        spade_activation: nn.Module = nn.ReLU,
        spade_activation_args: dict = {},
        spade_eps: float = 1e-5,
        custom_padding: nn.Module = lambda x, pdi: F.pad(x, pdi, "reflect"),
    ):
        super(SPADEGeneratorUnit, self).__init__()
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Standard deviation for Gaussian noise
        self.noise_std = noise_std
        # Activation function
        self.act = activation(**activation_args)
        # SPADE and convolution layers in the main path
        self.spade1 = SPADE(
            in_channels,
            mask_channels,
            spade_kernel_size,
            spade_activation,
            spade_activation_args,
            spade_eps,
            custom_padding,
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=0
        )
        self.spade2 = SPADE(
            out_channels,
            mask_channels,
            spade_kernel_size,
            spade_activation,
            spade_activation_args,
            spade_eps,
            custom_padding,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )
        # Skip connection
        if (in_channels == out_channels) and (not skip_spade):
            self.learned_skip = False
            self.skip_spade = None
            self.skip_conv = None
        elif (in_channels == out_channels) and skip_spade:
            self.learned_skip = True
            self.skip_spade = SPADE(
                in_channels,
                mask_channels,
                spade_kernel_size,
                spade_activation,
                spade_activation_args,
                spade_eps,
                custom_padding,
            )
            self.skip_conv = nn.Identity()
        else:
            self.learned_skip = True
            self.skip_spade = SPADE(
                in_channels,
                mask_channels,
                spade_kernel_size,
                spade_activation,
                spade_activation_args,
                spade_eps,
                custom_padding,
            )
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, mask, add_noise: bool):
        """
        Forward pass of the SPADEGeneratorUnit.

        Args:
            x (torch.Tensor): Input feature map. Shape: [N, C_in, H, W].
            mask (torch.Tensor): Input mask for spatial modulation. Shape: [N, M, H', W'].
            add_noise (bool, optional): Whether to add Gaussian noise. If None, defaults to self.training.

        Returns:
            torch.Tensor: The output tensor after processing. Shape: [N, C_out, H', W'].
        """
        # Noise injection
        if add_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        # Main path
        spade1_out = self.spade1(x, mask)
        relu1_out = self.act(spade1_out)
        relu1_padded = self.padding(relu1_out, self.padding_instruction)
        conv1_out = self.conv1(relu1_padded)
        spade2_out = self.spade2(conv1_out, mask)
        relu2_out = self.act(spade2_out)
        relu2_padded = self.padding(relu2_out, self.padding_instruction)
        conv2_out = self.conv2(relu2_padded)
        # Add the outputs of the main path and the skip connection
        if not self.learned_skip:
            out = conv2_out + x
        else:
            out = conv2_out + self.skip_conv(self.skip_spade(x, mask))
        return out
