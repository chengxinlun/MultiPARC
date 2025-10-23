import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool2d(nn.Module):
    def __init__(self, filter_1d, stride, custom_padding):
        super().__init__()
        # Padding
        filter_len = filter_1d.shape[0]
        self.padding_instruction = [
            filter_len // 2,
        ] * 4
        if filter_len % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        self.padding = custom_padding
        self.stride = stride
        # Filter
        filter_2d = filter_1d[:, None] * filter_1d[None, :]
        filter_2d /= torch.sum(filter_2d)
        filter_2d.requires_grad = False
        self.register_buffer("filter_2d", filter_2d[None, None, :, :])

    def forward(self, x):
        x = self.padding(x, self.padding_instruction)
        filter_2d = self.filter_2d.expand(x.shape[1], -1, -1, -1)
        return F.conv2d(x, filter_2d, stride=self.stride, groups=x.shape[1])


class BlurMaxPool2d(nn.Module):
    def __init__(self, stride, bp2d_filter_1d, custom_padding):
        super().__init__()
        self.dense_max = nn.MaxPool2d(kernel_size=stride, stride=1)
        self.pool = BlurPool2d(bp2d_filter_1d, stride, custom_padding)
        # Padding
        self.padding = custom_padding
        self.padding_instruction = [
            stride // 2,
        ] * 4
        if stride % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1

    def forward(self, x):
        x = self.padding(x, self.padding_instruction)
        out = self.pool(self.dense_max(x))
        return out
