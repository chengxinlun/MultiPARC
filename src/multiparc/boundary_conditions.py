import torch.nn as nn
import torch.nn.functional as F


class PaddingAll(nn.Module):
    def __init__(self, padding_mode, padding_value):
        super().__init__()
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, x, padding_instruction):
        return F.pad(
            x, padding_instruction, mode=self.padding_mode, value=self.padding_value
        )


class PaddingXY(nn.Module):
    def __init__(self, padding_mode, padding_value):
        super().__init__()
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, x, pis):
        x = F.pad(
            x,
            [pis[0], pis[1], 0, 0],
            mode=self.padding_mode[0],
            value=self.padding_value[0],
        )
        x = F.pad(
            x,
            [0, 0, pis[2], pis[3]],
            mode=self.padding_mode[1],
            value=self.padding_value[1],
        )
        return x
