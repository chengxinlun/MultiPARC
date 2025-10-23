import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class RK4(nn.Module):
    def __init__(self, use_checkpoint, **kwarg):
        super(RK4, self).__init__(**kwarg)
        self.use_checkpoint = use_checkpoint

    def forward(self, f, t, current, step_size):
        """
        RK4 integration. Fixed step, 4th order.

        Parameters
        ----------
        f: function, the function that returns time deriviative
        current: tensor, the current state and velocity variables
        step_size: float, integration step size

        Returns
        -------
        final_state: tensor with the same shape of ```current```, the next state and velocity varaibles
        update: tensor with the same shape of ```current```, the update in this step
        """
        # Compute k1
        if self.use_checkpoint:
            k1 = checkpoint(f, t, current, use_reentrant=False)
        else:
            k1 = f(t, current)
        # Compute k2
        inp_k2 = current + 0.5 * step_size * k1
        if self.use_checkpoint:
            k2 = checkpoint(f, t + 0.5 * step_size, inp_k2, use_reentrant=False)
        else:
            k2 = f(t + 0.5 * step_size, inp_k2)
        # Compute k3
        inp_k3 = current + 0.5 * step_size * k2
        if self.use_checkpoint:
            k3 = checkpoint(f, t + 0.5 * step_size, inp_k3, use_reentrant=False)
        else:
            k3 = f(t + 0.5 * step_size, inp_k3)
        # Compute k4
        inp_k4 = current + step_size * k3
        if self.use_checkpoint:
            k4 = checkpoint(f, t + step_size, inp_k4, use_reentrant=False)
        else:
            k4 = f(t + step_size, inp_k4)
        # Final
        update = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        final_state = current + step_size * update
        return final_state, update
