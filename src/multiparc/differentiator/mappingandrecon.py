import torch.nn as nn
from multiparc.utility.spade import SPADEGeneratorUnit
from multiparc.utility.resnet import ResNet
from multiparc.boundary_conditions import PaddingAll
from multiparc.utility.multires import MRResNet, MRConv2d


class MappingAndRecon(nn.Module):
    def __init__(
        self,
        n_base_features,
        n_mask_channel,
        output_channel,
        add_noise=True,
        spadegen_kernel_size=1,
        spadegen_activation=nn.LeakyReLU,
        spadegen_activation_args={"negative_slope": 0.2},
        spadegen_noise_std=0.05,
        spade_kernel_size=3,
        spade_activation=nn.ReLU,
        spade_activation_args={},
        spade_eps=1e-5,
        resnet_n_blocks=2,
        resnet_kernel_size=1,
        resnet_normalization=None,
        resnet_normalization_args={},
        resnet_activation=nn.ReLU,
        resnet_activation_args={},
        custom_padding=PaddingAll("reflect", 0),
    ):
        '''
        Module for combining implicit and explicit features to predict temporal deriviatives. Implicit and explicit features are first passed through a SPADE generator to combine both, with explicit features being the masks. The combined features are then passed through a ResNet to predict the temporal deriviatives.

        Args:
        n_base_features: int. Number of implicit features.
        n_mask_channel: int. Number of explicit features.
        output_channel: int. Number of output channels.
        add_noise: bool, optional, default ```True```. Whether to add noise during training or not.
        spadegen_kernel_size: int, optional, default ```1```. Kernel size of SPADE generator module.
        spadegen_activation: nn.Module, optional, default ```nn.LeakyReLU```. Activation function for SPADE generator module.
        spadegen_activation_args: dict, optional, default ```{"negative_slope": 0.2}```. Arguments to pass to the constructor of ```spadegen_activation```.
        spadegen_noise_std: float, optional, default ```0.05```. Standard deviation for the noise added.
        spade_kernel_size: int, optional, default ```3```. Kernel size within SPADE modules.
        spade_activation: nn.Module, optional, default ```nn.ReLU```. Activation function for SPADE module.
        spade_activation_args: dict, optional, default ```{}```. Arguments to pass to the constructor of ```spade_activation```.
        spade_esp: float, optional, default ```1e-5```. Small values to prevent division by zero.
        resnet_n_blocks: int, optional, default ```2```. Number of ResNetBlocks to include within the ResNet.
        resnet_kernel_size: int, optional, default ```1```. Kernel size of the ResNet.
        resnet_normalization: nn.Module, optional, default ```None```. Normalization layer within ResNet. Default value corresponds to no normalization.
        resnet_normalization_args: dict, optional, default ```{}```. Arguments to pass to the constructor of ```resnet_normalization```. Number of channels are automatically calulated and do not need to be supplied.
        resnet_activation: nn.Module, optional, default ```nn.ReLU```. Activation function for the ResNet.
        resnet_activation_args: dict, optional, default ```{}```. Arguments to pass to the constructor of ```resnet_activation```.
        custom_padding: nn.Module, optional, default ```PaddingAll("reflect", 0)```. Custom padding module for enforcement of boundary conditions.
        '''
        super(MappingAndRecon, self).__init__()
        self.add_noise = add_noise

        # Initialize SPADE generator unit
        self.spade = SPADEGeneratorUnit(
            in_channels=n_base_features,
            out_channels=n_base_features,
            mask_channels=n_mask_channel,
            kernel_size=spadegen_kernel_size,
            activation=spadegen_activation,
            activation_args=spadegen_activation_args,
            noise_std=spadegen_noise_std,
            spade_kernel_size=spade_kernel_size,
            spade_activation=spade_activation,
            spade_activation_args=spade_activation_args,
            spade_eps=spade_eps,
            custom_padding=custom_padding,
        )

        # Initialize ResNet block
        self.resnet = ResNet(
            in_channels=n_base_features,
            block_dimensions=[
                n_base_features,
            ]
            * resnet_n_blocks,
            kernel_size=resnet_kernel_size,
            normalization=resnet_normalization,
            normalization_args=resnet_normalization_args,
            activation=resnet_activation,
            activation_args=resnet_activation_args,
            pooling=None,
            pooling_args={},
            custom_padding=custom_padding,
        )
        # Final convolution layer
        self.conv_out = nn.Conv2d(
            in_channels=n_base_features,
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
        )

    def forward(self, dynamic_feature, advec_diff):
        """
        Forward pass of the MappingAndRecon.

        Args:
            dynamic_feature (torch.Tensor): Tensor of shape [N, C, H, W], dynamic features from the feature extraction network
            advec_diff (torch.Tensor): Tensor of shape [N, M, H, W], channel-concatenated advection and diffusion

        Returns:
            torch.Tensor: Output tensor of shape [N, output_channel, H, W]
        """
        spade_out = self.spade(dynamic_feature, advec_diff, self.add_noise)
        resnet_out = self.resnet(spade_out)
        conv_out = self.conv_out(resnet_out)

        return conv_out

    
class MRMAR(nn.Module):
    '''
    Multigrid Mapping and Reconstruction
    '''
    def __init__(self, 
                 n_base_features_lists: list[int], 
                 n_mask_channel: int, 
                 n_out_channel: int,
                 add_noise: bool,
                 sampling_factor: int, 
                 updown_mode: str,
                 spadegen_kernel_size=1,
                 spadegen_activation=nn.LeakyReLU,
                 spadegen_activation_args={"negative_slope": 0.2},
                 spadegen_noise_std=0.05,
                 spade_kernel_size=3,
                 spade_activation=nn.ReLU,
                 spade_activation_args={},
                 spade_eps=1e-5,
                 resnet_n_blocks=2,
                 resnet_kernel_size=1,
                 resnet_normalization=None,
                 resnet_normalization_args={},
                 resnet_activation=nn.ReLU,
                 resnet_activation_args={},
                 custom_padding=PaddingAll("reflect", 0),):
        super().__init__()
        self.add_noise = add_noise
        self.n_scales = len(n_base_features_lists)

        # One spade per scale
        all_the_spades = [SPADEGeneratorUnit(in_channels=each_scale,
                                             mask_channels=n_mask_channel,
                                             out_channels=each_scale,
                                             kernel_size=spadegen_kernel_size,
                                             activation=spadegen_activation,
                                             activation_args=spadegen_activation_args,
                                             noise_std=spadegen_noise_std,
                                             spade_kernel_size=spade_kernel_size,
                                             spade_activation=spade_activation,
                                             spade_activation_args=spade_activation_args,
                                             spade_eps=spade_eps,
                                             custom_padding=custom_padding,)
                          for each_scale in n_base_features_lists]
        self.spade = nn.ModuleList(all_the_spades)

        self.resnet = MRResNet(n_in_list=n_base_features_lists, 
            n_hidden_list=[n_base_features_lists,] * resnet_n_blocks,
            sampling_factor=sampling_factor, 
            updown_mode=updown_mode,
            kernel_size=resnet_kernel_size,
            normalization=resnet_normalization,
            normalization_args=resnet_normalization_args,
            activation=resnet_activation,
            activation_args=resnet_activation_args,
            custom_padding=custom_padding,)
        # Combine all the scales
        scale_combine = []
        for i in range(self.n_scales, 1, -1):
            scale_combine.append(MRConv2d(n_base_features_lists[:i], n_base_features_lists[:i-1], resnet_kernel_size, sampling_factor, updown_mode, custom_padding))
        self.scale_combine = nn.Sequential(*scale_combine)
        self.final_conv = nn.Conv2d(n_base_features_lists[0], n_out_channel, 1)

    def forward(self, dynamic_feature, advec_diff):
        spade_out = []
        for i in range(self.n_scales):
            spade_out.append(self.spade[i](dynamic_feature[i], advec_diff[i], self.add_noise))
        resnet_out = self.resnet(spade_out)
        single_out = self.scale_combine(resnet_out)
        conv_out = self.final_conv(single_out[0])
        return conv_out


class MRMARNoSpade(nn.Module):
    def __init__(self, 
                 n_base_features_lists: list[int], 
                 n_mask_channel: int, 
                 n_out_channel: int,
                 add_noise: bool,
                 sampling_factor: int, 
                 updown_mode: str,
                 instancenorm_args={"affine": False, "track_running_stats": False},
                 resnet_n_blocks=2,
                 resnet_kernel_size=1,
                 resnet_normalization=None,
                 resnet_normalization_args={},
                 resnet_activation=nn.ReLU,
                 resnet_activation_args={},
                 custom_padding=PaddingAll("reflect", 0),):
        all_the_norms = [nn.InstanceNorm2d(each_scale, **instancenorm_args) for each_scale in n_base_features_lists]
        self.norm = nn.ModuleList(all_the_norms)
        self.resnet = MRResNet(n_in_list=n_base_features_lists, 
            n_hidden_lists=[n_base_features_lists,] * resnet_n_blocks,
            sampling_factor=sampling_factory, 
            updown_mode=updown_mode,
            kernel_size=resnet_kernel_size,
            normalization=resnet_normalization,
            normalization_args=resnet_normalization_args,
            activation=resnet_activation,
            activation_args=resnet_activation_args,
            custom_padding=custom_padding,)
        # Combine all the scales
        scale_combine = []
        for i in range(self.n_scales, 1, -1):
            scale_combine.append(MRConv2d(n_base_features_lists[:i], n_base_features_lists[:i-1], resnet_kernel_size, sampling_factor, updown_mode, custom_padding))
        self.scale_combine = nn.Sequential(*scale_combine)
        self.final_conv = nn.Conv2d(n_base_features_lists[0], n_out_channel, 1)
    
    def forward(self, dynamic_feature, advec_diff):
        spade_out = []
        for i in range(self.n_scales):
            spade_out.append(self.norm[i](dynamic_feature[i]))
        resnet_out = self.resnet(spade_out)
        single_out = self.scale_combine(resnet_out)
        conv_out = self.final_conv(single_out[0])
        return conv_out
