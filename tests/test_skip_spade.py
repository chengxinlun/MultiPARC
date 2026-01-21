from multiparc.utility.spade import SPADE
from multiparc.utility.unet import UNet
from multiparc.utility.blurpool import BlurMaxPool2d
from multiparc.boundary_conditions import PaddingXY
from multiparc.differentiator.finitedifference import FiniteDifferenceGrad
from multiparc.differentiator.advection import AdvectionUpwind
from multiparc.differentiator.diffusion import Diffusion
from multiparc.differentiator.differentiator import ADRDifferentiator
from multiparc.integrator.rk4 import RK4
from multiparc.integrator.integrator import Integrator
from multiparc.PARCv2 import PARCv2

import torch
import torch.nn as nn
from pathlib import Path
import os
import random


def test_baseline_parcv2():
    torch.manual_seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    test_dir = Path(__file__).parent
    bce = PaddingXY(["circular", "reflect"], [0.0, 0.0])
    depth_ratio = 1.54
    unet = UNet(
        [
            64,
            int(64 * depth_ratio),
            int(64 * depth_ratio * depth_ratio),
            int(64 * depth_ratio * depth_ratio * depth_ratio),
        ],
        [
            int(64 * depth_ratio * depth_ratio * depth_ratio),
            int(64 * depth_ratio * depth_ratio * depth_ratio),
        ],
        5,
        64,
        normalization=None,
        normalization_args={},
        activation=nn.LeakyReLU,
        activation_args={"negative_slope": 0.2},
        pooling=nn.MaxPool2d,
        pooling_args={"kernel_size": 2},
        custom_padding=bce,
    ).cuda()
    # Numerical schemes
    grad_ldiff = FiniteDifferenceGrad(
        filter_1d=torch.tensor([-1.0, 1.0], dtype=torch.float32),
        right_bottom=False,
        custom_padding=bce,
    ).cuda()
    grad_rdiff = FiniteDifferenceGrad(
        filter_1d=torch.tensor([-1.0, 1.0], dtype=torch.float32),
        right_bottom=True,
        custom_padding=bce,
    ).cuda()
    adv = AdvectionUpwind(grad_ldiff, grad_rdiff).cuda()
    dif = Diffusion().cuda()
    # MAR
    mar_args = {
        "add_noise": False,
        "spadegen_kernel_size": 1,
        "spadegen_activation": nn.LeakyReLU,
        "spadegen_activation_args": {"negative_slope": 0.2},
        "spadegen_noise_std": 0.05,
        "spadegen_skip_spade": True,
        "spade_kernel_size": 3,
        "spade_activation": nn.ReLU,
        "spade_activation_args": {},
        "spade_eps": 1e-5,
        "resnet_n_blocks": 2,
        "resnet_kernel_size": 1,
        "resnet_normalization": None,
        "resnet_normalization_args": {},
        "resnet_activation": nn.ReLU,
        "resnet_activation_args": {},
        "custom_padding": bce,
    }
    channel_instructions = {
        0: {"c": True},  # 0: Constant channel
        1: {"a": [1], "r": True},  # 1: Adv(1) + Reaction
        2: {"a": [2], "r": True},  # 2: Adv(2) + Reaction
        (3, 4): {"a": [3, 4], "r": True},  # 3,4: Adv(3, 4) + Reaction
    }
    # Differentiator
    diff = ADRDifferentiator(64, channel_instructions, unet, adv, dif, mar_args).cuda()
    # Integrator
    rk4_int = RK4(use_checkpoint=False).cuda()
    rk4_int = Integrator(False, rk4_int)
    # Baseline PARCv2
    parc_model = PARCv2(diff, rk4_int).cuda()
    # No skip spade
    for n, m in parc_model.named_modules():
        if n.endswith(".spade"):
            assert isinstance(m.skip_conv, nn.Identity)
            assert isinstance(m.skip_spade, SPADE)
            print(n + " has learnt skip connections: " + str(m.learned_skip))
    # Clean up
    del parc_model
    torch.cuda.empty_cache()