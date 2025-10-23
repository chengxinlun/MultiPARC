import torch
import torch.nn as nn
from multiparc.differenitator.mappingandrecon import MappingAndRecon, MRMAR, MRMARNoSpade
from multiparc.utility.resnet import ResNet
from multiparc.utility.multires import _updown_mode_dict


class ChannelDifferentiator(nn.Module):
    def __init__(
        self,
        adv_idx,
        dif_idx,
        feature_channels,
        out_channels,
        advection,
        diffusion,
        mar_args,
    ):
        super().__init__()
        self.adv_idx = adv_idx
        self.dif_idx = dif_idx
        self.n_explicit_features = self.adv_idx.nelement() + self.dif_idx.nelement()
        # Advection
        if self.adv_idx.nelement() != 0:
            self.adv = advection
        else:
            self.adv = None
        # Diffusion
        if self.dif_idx.nelement() != 0:
            self.dif = diffusion
        else:
            self.dif = None
        # MAR
        if self.n_explicit_features == 0:
            # No explicit features: normalize the feature channels and send it to a resnet regressor
            self.mar = nn.Sequential(
                nn.InstanceNorm2d(
                    feature_channels, affine=False, track_running_stats=False
                ),
                ResNet(
                    in_channels=feature_channels,
                    block_dimensions=[
                        feature_channels,
                    ]
                    * mar_args["resnet_n_blocks"],
                    kernel_size=mar_args["resnet_kernel_size"],
                    normalization=mar_args["resnet_normalization"],
                    normalization_args=mar_args["resnet_normalization_args"],
                    activation=mar_args["resnet_activation"],
                    activation_args=mar_args["resnet_activation_args"],
                    pooling=None,
                    pooling_args={},
                    custom_padding=mar_args["custom_padding"],
                ),
                nn.Conv2d(feature_channels, out_channels, kernel_size=1, padding=0),
            )
        else:
            # With explicit features: use SPADE and sent it to a resnet regressor
            self.mar = MappingAndRecon(
                feature_channels, self.n_explicit_features, out_channels, **mar_args
            )

    def forward(self, x, features):
        # Explicit featuers
        explicit_features = []
        if self.adv_idx.nelement() != 0:
            explicit_features.append(
                self.adv(torch.index_select(x, 1, self.adv_idx), x[:, -2:, :, :])
            )
        if self.dif_idx.nelement() != 0:
            explicit_features.append(self.dif(torch.index_select(x, 1, self.dif_idx)))
        # Recombine
        if self.n_explicit_features == 0:
            out = self.mar(features)
        else:
            out = self.mar(features, torch.cat(explicit_features, dim=1))
        return out


class ADRDifferentiator(nn.Module):
    def __init__(
        self,
        n_fe_features,
        channel_instructions,
        feature_extraction,
        advection,
        diffusion,
        mar_args,
        device="cuda",
    ):
        '''
        Advection-Diffusion-Reaction differentiator
    
        Args:
            n_fe_features: int. Number of output features from feature extractor network
            channel_instructions: dict. Governs how the implicit and explicit features are combined, which explicit features to include and which channel(s) to output to. See tutorial for details.
            features_extraction: nn.Module. Feature extraction model. Output must have the shape of ```(b, n_fe_features, y, x)``` where ```b``` is the batch size, ```y``` and ```x``` the spacial size of the input.
            advection: nn.Module. Module for calculating advection. Signature of forward function must follow ```differentiator.advection.Advection```.
            diffusion: nn.Module. Module for calculating diffusion. Signature of forward function must follow ```differentiator.diffusion.Diffusion```.
            mar_args: dict, arguments to pass to the constructor of modules for combinining implicit and explicit features.
            device: str, optional, default ```cuda```. Device where the models stay.
        '''
        super().__init__()
        self.feature_extraction = feature_extraction
        # Parsing the channel instructions
        self.out_idx = []
        self.modules_list = nn.ModuleList()
        for each_out, in_instruction in channel_instructions.items():
            if isinstance(each_out, tuple):
                out_channels = len(each_out)
                self.out_idx.append(each_out)
            else:
                out_channels = 1
                self.out_idx.append((each_out,))
            if ("c" in in_instruction) and (in_instruction["c"]):
                # Case 1: constant channel
                self.modules_list.append(None)
            else:
                # Case 2: parsing adr
                if "r" not in in_instruction:
                    raise ValueError(
                        "Feature extraction is required for non-constant channels."
                    )
                adv_idx = in_instruction.get("a", [])
                dif_idx = in_instruction.get("d", [])
                self.modules_list.append(
                    ChannelDifferentiator(
                        torch.tensor(adv_idx, dtype=int).to(device),
                        torch.tensor(dif_idx, dtype=int).to(device),
                        n_fe_features,
                        out_channels,
                        advection,
                        diffusion,
                        mar_args,
                    )
                )

    def forward(self, t, current):
        """
        Forward of differentiator.

        Args:
            t: float, float scalar for current time
            current: 4-d tensor of Float with shape (batch_size, channels, y, x), the current state and velocity variables

        Returns:
            t_dot: 4-d tensor of Float with the same shape as ```current```, the predicted temporal deriviatives for all channels
        """
        dynamic_features = self.feature_extraction(current)
        t_dot = torch.zeros_like(current)
        for each_out, each_module in zip(self.out_idx, self.modules_list):
            if each_module is not None:
                t_dot_channel = each_module(current, dynamic_features)
                # Replace the channels with each_out
                for i, idx in enumerate(each_out):
                    t_dot[:, idx, :, :] = t_dot_channel[:, i, :, :]
        return t_dot
    
    
class MGChannelDifferentiator(nn.Module):
    def __init__(
        self,
        adv_idx,
        dif_idx,
        n_feature_channels_lists,
        n_out_channels,
        advection,
        diffusion,
        mar_spade_args,
        mar_no_spade_args,
    ):
        super().__init__()
        self.adv_idx = adv_idx
        self.dif_idx = dif_idx
        self.n_explicit_features = self.adv_idx.nelement() + self.dif_idx.nelement()
        # Advection
        if self.adv_idx.nelement() != 0:
            self.adv = advection
        else:
            self.adv = None
        # Diffusion
        if self.dif_idx.nelement() != 0:
            self.dif = diffusion
        else:
            self.dif = None
        # MAR
        if self.n_explicit_features == 0:
            # No explicit features: normalize the feature channels and send it to a resnet regressor
            self.mar = MGMARNoSpade(n_feature_channels_lists, self.n_explicit_features, n_out_channels, **mar_nospade_args)
        else:
            # With explicit features: use SPADE and sent it to a resnet regressor
            self.mar = MGMAR(
                n_feature_channels_lists, self.n_explicit_features, n_out_channels, **mar_spade_args
            )

    def forward(self, x, features):
        # Explicit featuers
        explicit_features = []
        # Multigrid
        for each_x in x:
            ef_scale = []
            if self.adv_idx.nelement() != 0:
                ef_scale.append(
                    self.adv(torch.index_select(each_x, 1, self.adv_idx), each_x[:, -2:, :, :])
                )
            if self.dif_idx.nelement() != 0:
                ef_scale.append(self.dif(torch.index_select(each_x, 1, self.dif_idx)))
            explicit_features.append(torch.cat(ef_scale, dim=1))
        # Recombine
        out = self.mar(features, explicit_features)
        return out    

    
class MGADRDifferentiator(nn.Module):
    def __init__(
        self,
        n_fe_features_list,
        channel_instructions,
        feature_extraction,
        advection,
        diffusion,
        mar_spade_args,
        mar_nospade_args,
        device="cuda",
    ):
        '''
        Multi-resolution Advection-Diffusion-Reaction differentiator
    
        Args:
            n_fe_features_list: list[int]. Number of output features from multi-resolution feature extractor network
            channel_instructions: dict. Governs how the implicit and explicit features are combined, which explicit features to include and which channel(s) to output to. See tutorial for details.
            features_extraction: nn.Module. Multi-resolution feature extraction model.
            advection: nn.Module. Module for calculating advection. Signature of forward function must follow ```differentiator.advection.Advection```.
            diffusion: nn.Module. Module for calculating diffusion. Signature of forward function must follow ```differentiator.diffusion.Diffusion```.
            mar_spade_args: dict. Arguments to pass to the constructor of modules for combinining implicit and explicit features.
            mar_nospade_args: dict. Arguments to pass to the constructor of modules for with only implicit features.
            device: str, optional, default ```cuda```. Device where the models stay.
        '''
        super().__init__()
        self.n_scales = len(n_fe_features_list)
        self.feature_extraction = feature_extraction
        self.downsampler = _updown_mode_dict[mar_spade_args["updown_mode"]][0](mar_spade_args["sampling_factor"])
        # Parsing the channel instructions
        self.out_idx = []
        self.modules_list = nn.ModuleList()
        for each_out, in_instruction in channel_instructions.items():
            if isinstance(each_out, tuple):
                out_channels = len(each_out)
                self.out_idx.append(each_out)
            else:
                out_channels = 1
                self.out_idx.append((each_out,))
            if ("c" in in_instruction) and (in_instruction["c"]):
                # Case 1: constant channel
                self.modules_list.append(None)
            else:
                # Case 2: parsing adr
                if "r" not in in_instruction:
                    raise ValueError(
                        "Feature extraction is required for non-constant channels."
                    )
                adv_idx = in_instruction.get("a", [])
                dif_idx = in_instruction.get("d", [])
                self.modules_list.append(
                    MGChannelDifferentiator(
                        torch.tensor(adv_idx, dtype=int).to(device),
                        torch.tensor(dif_idx, dtype=int).to(device),
                        n_fe_features_list,
                        out_channels,
                        advection,
                        diffusion,
                        mar_spade_args,
                        mar_nospade_args
                    )
                )

    def forward(self, t, current):
        """
        Forward of differentiator.

        Args:
            t: float, float scalar for current time
            current: 4-d tensor of Float with shape (batch_size, channels, y, x), the current state and velocity variables

        Returns:
            t_dot: 4-d tensor of Float with the same shape as ```current```, the predicted temporal deriviatives for all channels
        """
        # Get the multi-grid
        mg_current = [current]
        for _ in range(1, self.n_scales):
            mg_current.append(self.downsampler(mg_current[-1]))
        # Feature extraction
        dynamic_features = self.feature_extraction(mg_current)
        # Temporal deriviative
        t_dot = torch.zeros_like(current)
        for each_out, each_module in zip(self.out_idx, self.modules_list):
            if each_module is not None:
                t_dot_channel = each_module(mg_current, dynamic_features)
                # Replace the channels with each_out
                for i, idx in enumerate(each_out):
                    t_dot[:, idx, :, :] = t_dot_channel[:, i, :, :]
        return t_dot
