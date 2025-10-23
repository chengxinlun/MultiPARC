import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResNetBlock(nn.Module):
    """
    Residual Block for ResNet with support for changing feature dimensions.

    x --> Conv2d --> ReLU --> Conv2d --> ReLU
       |                              |
       -------- Identity/Conv2d -------

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        normalization (nn.Module or None): Normalization
        normalization_args (dict): Args to pass to normalization during initialization.
        activation (nn.Module): Activation function.
        activation_args (dict): Args to pass to activation function during initialization.
        custom_padding (callable): User customized padding for boundary condition enforcement.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        normalization: nn.Module,
        normalization_args: dict,
        activation: nn.Module,
        activation_args: dict,
        custom_padding: nn.Module,
    ):
        super(ResNetBlock, self).__init__()
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Convolutions
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=0
        )
        self.act = activation(**activation_args)
        if normalization is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = normalization(out_channels, **normalization_args)
            self.norm2 = normalization(out_channels, **normalization_args)
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.padding(x, self.padding_instruction)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.padding(out, self.padding_instruction)
        out = self.conv2(out)
        skip = self.skip_conv(x)
        out = self.act(self.norm2(out + skip))
        return out


class ResNet(nn.Module):
    """
    ResNet model consisting of multiple residual blocks with varying feature dimensions.

    The model initializes with a convolutional layer and iteratively adds residual blocks
    based on the provided `block_dimensions`. Optionally, it applies max pooling after
    each residual block to reduce spatial dimensions. Model outputs pass through an activation
    funtion at the end, so do not use for regression purposes without attaching an extra
    convolution layer.

    Args:
        in_channels (int): Number of input channels.
        block_dimensions (List[int]): List specifying the number of feature channels for each residual block.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        normalization (nn.Module or None, optional): Normalization layer. Default is None.
        normalization_args (dict, optional): Args to pass to normalization layer during initialization. Default is an empty dictionary.
        activation (nn.Module, optional): Activation function. Default is nn.ReLU.
        activation_args (dict, optional): Args to pass to activation function during initialization. Default is an empty dictionary.
        pooling (nn.Module, optional): Pooling. Default is None.
        pooling_args (dict, optional): Args to pass to pooling. Default is an empty dictionary.
        custom_padding: nn.Module
    """

    def __init__(
        self,
        in_channels: int,
        block_dimensions: List[int],
        kernel_size: int = 3,
        normalization: nn.Module = None,
        normalization_args: dict = {},
        activation: nn.Module = nn.ReLU,
        activation_args: dict = {},
        pooling: nn.Module = None,
        pooling_args: dict = {},
        custom_padding: nn.Module = lambda x, pdi: F.pad(x, pdi, "reflect"),
    ):
        super(ResNet, self).__init__()
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Double convolution + ReLU
        self.conv1 = nn.Conv2d(
            in_channels, block_dimensions[0], kernel_size=kernel_size, padding=0
        )
        self.conv2 = nn.Conv2d(
            block_dimensions[0], block_dimensions[0], kernel_size=kernel_size, padding=0
        )
        self.act = activation(**activation_args)
        if normalization is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = normalization(block_dimensions[0], **normalization_args)
            self.norm2 = normalization(block_dimensions[0], **normalization_args)
        # The blocks
        module_list = []
        for i in range(len(block_dimensions)):
            if i == 0:
                in_channels = block_dimensions[0]
            else:
                in_channels = block_dimensions[i - 1]
            res_block = ResNetBlock(
                in_channels=in_channels,
                out_channels=block_dimensions[i],
                kernel_size=kernel_size,
                normalization=normalization,
                normalization_args=normalization_args,
                activation=activation,
                activation_args=activation_args,
                custom_padding=custom_padding,
            )
            module_list.append(res_block)
            if pooling is not None:
                module_list.append(pooling(**pooling_args))
        self.path = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Double convolution
        out = self.padding(x, self.padding_instruction)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.padding(out, self.padding_instruction)
        out = self.act(self.norm2(self.conv2(out)))
        # Blocks
        out = self.path(out)
        return out


class ResNetRegressor(nn.Module):
    """
    ResNet (no pooling) + 1x1 convolution for regression purposes.
    """

    def __init__(
        self,
        in_channels,
        block_dimensions,
        kernel_size,
        out_channels,
        normalization,
        normalization_args,
        activation,
        activation_args,
        custom_padding,
    ):
        super().__init__()
        self.resnet = ResNet(
            in_channels,
            block_dimensions,
            kernel_size,
            normalization,
            normalization_args,
            activation,
            activation_args,
            None,
            {},
            custom_padding,
        )
        self.final_conv = nn.Conv2d(block_dimensions[-1], out_channels, 1)

    def forward(self, x):
        return self.final_conv(self.resnet(x))