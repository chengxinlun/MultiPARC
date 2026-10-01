import torch
from torch import nn


class Integrator(nn.Module):
    def __init__(
        self,
        clip: bool,
        num_int: nn.Module,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        detach_intermediate: bool = False,
    ):
        """
        Constructor of integrator

        Args:
            clip: bool, whether to clip value or not. Note that clip occurs before the numerical integrator call
            num_int: nn.module, numerical integrator. Forward function must have the following signature: ```(f, t0, current, delta_t)```, where ```f``` is the differentiator, ```t0``` is current time, ```current``` is current state, ```delta_t``` is time step.
            clip_min: float, optional, default 0.0. Minimal value of the clip function.
            clip_max: float, optional, default 1.0. Maximal value of the clip function.
            detach_intermediate: bool, optional, default False. Whether to detach the intermediate states between integrator stepping.
        """
        super().__init__()
        self.clip = clip
        self.numerical_integrator = num_int
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.detach_intermediate = detach_intermediate

    def forward(self, f, ic, t0, t1):
        """
        Forward of Integrator. It will clip the current state and velocity variable (if necessary), go through the numerical integrator and then datadriven integrator.

        Args
            f: callable, callable that returns time derivative
            ic: torch.tensor of shape (b, c, y, x), the initial condition
            t0: float, starting time
            t1: torch.tensor of shape (ts,), the time points to predict at

        Returns
            res: torch.tensor of shape (ts, b, c, y, x), the predicted state and velocity variables at each time in t1
        """
        all_time = torch.cat([t0.unsqueeze(0), t1])
        n_channel = ic.shape[1]
        n_state_var = n_channel - 2
        res = []
        current = ic
        for ts in range(1, all_time.shape[0]):
            if self.clip:
                current = torch.clamp(current, self.clip_min, self.clip_max)
            if self.detach_intermediate:
                current = current.detach()
            # Numerical integrator
            current, update = self.numerical_integrator(
                f, all_time[ts - 1], current, all_time[ts] - all_time[ts - 1]
            )
            res.append(current.unsqueeze(0))
        res = torch.cat(res, 0)
        return res
