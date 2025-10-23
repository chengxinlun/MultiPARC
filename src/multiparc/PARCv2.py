import torch.nn as nn


class PARCv2(nn.Module):
    def __init__(self, differentiator, integrator, **kwargs):
        """
        Constructor of PARCv2.

        Args:
            differentiator: nn.Module, the differentiator
            integrator: nn.Module, the numerical and data-driven (if necessary) integrator
            **kwargs: other parameters that will be passed onto torch.nn.Module
        """
        super(PARCv2, self).__init__(**kwargs)

        self.differentiator = differentiator
        self.integrator = integrator

    def freeze_differentiator(self):
        """
        A convenient function to freeze the differentiator
        """
        for parameter in self.differentiator.parameters():
            parameter.requires_grad = False
        self.differentiator.eval()

    def forward(self, ic, t0, t1):
        """
        Forward of PARCv2. Essentially a call to the integrator with the differentiator.
        Note that we assume all samples within a batch are of same physical time.

        Args:
            ic: torch.tensor of shape (b, c, y, x), initial condition.
            t0: float, starting time of the initial condition
            t1: torch.tensor of shape (ts,), time point that PARCv2 will predict on

        Returns:
            res: torch.tensor of shape (ts, b, c, y, x), predicted sequences at each time point in t1
        """
        return self.integrator(self.differentiator, ic, t0, t1)
