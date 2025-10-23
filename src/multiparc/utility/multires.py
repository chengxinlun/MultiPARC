import torch
import torch.nn as nn


_updown_mode_dict = {
    "avg-bilinear": [nn.AvgPool2d, "bilinear"],
    "max-nearest": [nn.MaxPool2d, "nearest"]
}


class MRConv2d(nn.Module):
    '''
    Multigrid convolution. Nearest scales are upsampled/downsampled and concatenated before convolution. This convolution module will also take care of any padding.
    '''
    def __init__(self, n_in_list, n_out_list, kernel_size, sampling_factor, updown_mode, custom_padding):
        super().__init__()
        # Padding instructions
        self.padding = custom_padding
        self.padding_instruction = [
            kernel_size // 2,
        ] * 4
        if kernel_size % 2 == 0:
            self.padding_instruction[0] -= 1
            self.padding_instruction[2] -= 1
        # Different scales
        self.n_scales_in = len(n_in_list)
        assert self.n_scales_in > 1
        self.n_scales_out = len(n_out_list)
        # Down/up-sampler
        assert updown_mode in _updown_mode_dict
        self.upsampler = nn.Upsample(scale_factor=sampling_factor, mode=_updown_mode_dict[updown_mode][1])
        self.downsampler = _updown_mode_dict[updown_mode][0](sampling_factor)
        # The convolution layers
        convs = []
        n_in_list_padded = [0, *n_in_list, 0]
        for i in range(self.n_scales_out):
            convs.append(nn.Conv2d(n_in_list_padded[i] + n_in_list_padded[i+1] + n_in_list_padded[i+2], 
                                   n_out_list[i], kernel_size, padding=0))
        self.conv_list = nn.ModuleList(convs)
    
    def forward(self, mr_x):
        mr_out = []
        # Finest scale
        conv_in = torch.cat([mr_x[0], self.upsampler(mr_x[1])], 1)
        conv_in = self.padding(conv_in, self.padding_instruction)
        mr_out.append(self.conv_list[0](conv_in))
        # Subsequent scales
        if self.n_scales_out < self.n_scales_in:
            n_sub = self.n_scales_out
        else:
            n_sub = self.n_scales_out - 1
        for i in range(1, n_sub):
            conv_in = torch.cat([self.downsampler(mr_x[i-1]), mr_x[i], self.upsampler(mr_x[i+1])], 1)
            conv_in = self.padding(conv_in, self.padding_instruction)
            mr_out.append(self.conv_list[i](conv_in))
        # Coarest scale (if exists)
        if self.n_scales_out == self.n_scales_in:
            conv_in = torch.cat([self.downsampler(mr_x[self.n_scales_out-2]), mr_x[self.n_scales_out-1]], 1)
            conv_in = self.padding(conv_in, self.padding_instruction)
            mr_out.append(self.conv_list[self.n_scales_out-1](conv_in))
        return mr_out

    
class MRMap(nn.Module):
    '''
    Multigrid mapping. A convenient function for [f_i(x_i) for i in range(n_scales)]. Therefore, there is no information exchange between scales.
    '''
    def __init__(self, model_list, single):
        super().__init__()
        self.single = single
        if self.single:
            assert len(model_list) == 1
        self.model_list = nn.ModuleList(model_list)
    
    def forward(self, mr_x):
        if self.single:
            return [self.model_list[0](each) for each in mr_x]
        else:
            return [self.model_list[i](mr_x[i]) for i in range(len(mr_x))]

        
class MRResNetBlock(nn.Module):
    '''
    Multigrid ResNetBlock.
    '''
    def __init__(self, n_in_list, n_hidden_list, n_out_list, kernel_size, sampling_factor, updown_mode, normalization, normalization_args, activation, activation_args, custom_padding):
        super().__init__()
        # Use MRConv2d to compare different scales
        assert len(n_in_list) == len(n_hidden_list)
        assert len(n_in_list) == len(n_out_list)
        self.n_scales = len(n_in_list)
        # Convs
        self.conv1 = MRConv2d(n_in_list, n_hidden_list, kernel_size, sampling_factor, updown_mode, custom_padding)
        self.conv2 = MRConv2d(n_hidden_list, n_out_list, kernel_size, sampling_factor, updown_mode, custom_padding)
        # Norms: seperate per scale
        if normalization is None:
            self.norm1 = MRMap([nn.Identity()], single=True)
            self.norm2 = MRMap([nn.Identity()], single=True)
        else:
            self.norm1 = MRMap([normalization(each_scale, **normalization_args) for each_scale in n_hidden_list])
            self.norm2 = MRMap([normalization(each_scale, **normalization_args) for each_scale in n_out_list])
        # Activation
        self.act = MRMap([activation(**activation_args)], single=True)
        # Skip connection
        skip_list = []
        for i in range(self.n_scales):
            if n_in_list[i] == n_out_list[i]:
                skip_list.append(nn.Identity())
            else:
                skip_list.append(nn.Conv2d(n_in_list[i], n_out_list[i], 1))
        self.skip = MRMap(skip_list, False)
        
    def forward(self, mr_x):
        mr_out = self.act(self.norm1(self.conv1(mr_x)))
        mr_out = self.norm2(self.conv2(mr_out))
        mr_skip = self.skip(mr_x)
        for i in range(self.n_scales):
            mr_out[i] += mr_skip[i]
        mr_out = self.act(mr_out)
        return mr_out

    
class MRResNet(nn.Module):
    '''
    Multigrid ResNet
    '''
    def __init__(self, n_in_list, n_hidden_list, kernel_size, sampling_factor, updown_mode, normalization, normalization_args, activation, activation_args, custom_padding):
        super().__init__()
        self.n_scales = len(n_in_list)
        # Convs
        self.conv1 = MRConv2d(n_in_list, n_hidden_list[0], kernel_size, sampling_factor, updown_mode, custom_padding)
        self.conv2 = MRConv2d(n_hidden_list[0], n_hidden_list[0], kernel_size, sampling_factor, updown_mode, custom_padding)
        # Norms: seperate per scale
        if normalization is None:
            self.norm1 = MRMap([nn.Identity()], single=True)
            self.norm2 = MRMap([nn.Identity()], single=True)
        else:
            self.norm1 = MRMap([normalization(each_scale, **normalization_args) for each_scale in n_hidden_list[0]])
            self.norm2 = MRMap([normalization(each_scale, **normalization_args) for each_scale in n_hidden_list[0]])
        # Activation
        self.act = MRMap([activation(**activation_args)], single=True)
        # Blocks
        block_list = []
        for i in range(len(n_hidden_list)):
            if i == 0:
                block_in_list = n_hidden_list[0]
            else:
                block_in_list = n_hidden_list[i-1]
            block_list.append(MRResNetBlock(block_in_list, n_hidden_list[i], n_hidden_list[i], kernel_size, sampling_factor, updown_mode, normalization, normalization_args, activation, activation_args, custom_padding))
        self.blocks = nn.Sequential(*block_list)
    
    def forward(self, mr_x):
        # Double convolution
        mr_out = self.act(self.norm1(self.conv1(mr_x)))
        mr_out = self.act(self.norm2(self.conv2(mr_out)))
        # Blocks
        mr_out = self.blocks(mr_out)
        return mr_out
