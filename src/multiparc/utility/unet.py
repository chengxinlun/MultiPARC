import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDownBlock(nn.Module):
    """
    U-Net Downsampling Block.

    Performs two convolutional operations followed by a pooling operation
    to reduce the spatial dimensions of the input tensor while increasing the
    number of feature channels. The output would be the tensor after the pooling and before the pooling operation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after convolution.
        kernel_size (int): Size of the convolutional kernels.
        normalization (nn.Module or None): Normalization layer.
        normalization_args (dict): Args for normalization layer.
        activation (nn.Module): Activation function.
        activation_args (dict): Args for activation function.
        pooling (nn.Module): Pooling.
        pooling_args (dict): Args for pooling.
        custom_padding (nn.Module): Custom padding module for boundary condition enforcement
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        normalization,
        normalization_args,
        activation,
        activation_args,
        pooling,
        pooling_args,
        custom_padding,
    ):
        super(UNetDownBlock, self).__init__()
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
        # Pooling
        self.pool = pooling(**pooling_args)

    def forward(self, x):
        x = self.padding(x, self.padding_instruction)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.padding(x, self.padding_instruction)
        x_features = self.act(self.norm2(self.conv2(x)))
        x = self.pool(x_features)
        return x, x_features


class UNetUpBlock(nn.Module):
    """
    U-Net Upsampling Block.

    Performs upsampling using a transposed convolutional layer, optionally concatenates
    the corresponding skip connection from the downsampling path, and applies two
    convolutional operations to refine the features.

    Args:
        in_channels (int): Number of input channels from the previous layer.
        out_channels (int): Number of output channels after convolution.
        skip_channels (int): Number of channels from the skip connection.
        kernel_size (int): Size of the convolutional kernels.
        normalization (nn.Module or None): Normalization layer.
        normalization_args (dict): Args for normalization layer.
        activation (nn.Module): Activation function.
        activation_args (dict): Args for activation function.
        custom_padding (nn.Module): Custom padding module for boundary condition enforcement
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        normalization,
        normalization_args,
        activation,
        activation_args,
        custom_padding,
    ):
        super(UNetUpBlock, self).__init__()
        # Upsampling
        self.upConv = nn.Upsample(scale_factor=2, mode="bilinear")
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Convolutions
        conv_in_channels = in_channels + skip_channels
        self.conv1 = nn.Conv2d(
            conv_in_channels, out_channels, kernel_size=kernel_size, padding=0
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

    def forward(self, x, skip_connection):
        x = self.upConv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.padding(x, self.padding_instruction)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.padding(x, self.padding_instruction)
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """
    U-Net Model.

    Constructs a U-Net architecture with customizable depth and feature dimensions.
    Supports selective use of skip connections and concatenation in the upsampling path.

    Args:
        block_dimensions (list of int): List of feature dimensions for each block.
        bottleneck_dimensions(list of int): List of feature dimensions for each bottleneck layer.
        output_channels (int): Number of output channels of the final layer.
        kernel_size (int, optional): Size of the convolutional kernels. Default is 3.
        normalization (nn.Module or None, optional): Normalization layer. Default is None.
        normalization_args (dict, optional): Args for normalization layer. Default is an empty dictionary.
        activation (nn.Module, optional): Activation function. Default is nn.LeakyReLU.
        activation_args (dict, optional): Args for activation function. Default is {"negative_slope": 0.2}.
        pooling (nn.Module, optional): Pooling. Default is nn.MaxPool2d.
        pooling_args (dict, optional): Args for pooling. Default is {"kernel_size": 2}.
        custom_padding (callable, optional): Custom padding module for boundary condition enforcement. Default is reflect padding at all boundaries
    """

    def __init__(
        self,
        block_dimensions,
        bottleneck_dimensions,
        in_channels,
        out_channels,
        kernel_size=3,
        normalization=None,
        normalization_args={},
        activation=nn.LeakyReLU,
        activation_args={"negative_slope": 0.2},
        pooling=nn.MaxPool2d,
        pooling_args={"kernel_size": 2},
        custom_padding=lambda x, pdi: F.pad(x, pdi, "reflect"),
    ):
        super(UNet, self).__init__()
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Downsampling blocks
        self.down_list = nn.ModuleList()
        skip_channels = []
        for i, each_down in enumerate(block_dimensions):
            if i == 0:
                cin = in_channels
            else:
                cin = block_dimensions[i - 1]
            self.down_list.append(
                UNetDownBlock(
                    cin,
                    each_down,
                    kernel_size,
                    normalization,
                    normalization_args,
                    activation,
                    activation_args,
                    pooling,
                    pooling_args,
                    custom_padding,
                )
            )
            skip_channels.append(each_down)
        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for i, each_neck in enumerate(bottleneck_dimensions):
            if i == 0:
                cin = block_dimensions[-1]
            else:
                cin = bottleneck_dimensions[i - 1]
            # Neckpiece
            neckpiece = []
            neckpiece.append(
                nn.Conv2d(cin, each_neck, kernel_size=kernel_size, padding=0)
            )
            if normalization is None:
                neckpiece.append(nn.Identity())
            else:
                neckpiece.append(normalization(each_neck, **normalization_args))
            neckpiece.append(activation(**activation_args))
            self.bottleneck.append(nn.Sequential(*neckpiece))
        # Upsampling blocks
        self.up_list = nn.ModuleList()
        block_dimensions = block_dimensions[::-1]
        skip_channels = skip_channels[::-1]
        for i, each_up in enumerate(block_dimensions):
            if i == 0:
                if len(bottleneck_dimensions) != 0:
                    cin = bottleneck_dimensions[-1]
                else:
                    cin = block_dimensions[0]
            else:
                cin = block_dimensions[i - 1]
            self.up_list.append(
                UNetUpBlock(
                    cin,
                    each_up,
                    skip_channels[i],
                    kernel_size,
                    normalization,
                    normalization_args,
                    activation,
                    activation_args,
                    custom_padding,
                )
            )
        # Final conv
        self.final_conv = nn.Conv2d(
            block_dimensions[-1], out_channels, kernel_size=1, padding=0
        )

    def forward(self, x):
        skip_connections = []
        # Downsampling block
        for each_downblock in self.down_list:
            x, x_skip = each_downblock(x)
            skip_connections.append(x_skip)
        # Bottleneck
        for each_bottleneck in self.bottleneck:
            x = self.padding(x, self.padding_instruction)
            x = each_bottleneck(x)
        # Upsampling block
        skip_connections = skip_connections[::-1]
        for i, each_upblock in enumerate(self.up_list):
            x = each_upblock(x, skip_connections[i])
        # Final conv
        x = self.final_conv(x)
        return x
