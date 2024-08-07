import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f K' % (num_params / 1e3))


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """ Conv. with activation. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############

class MuLUTConvUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, out_c=1, dense=True):
        super(MuLUTConvUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, out_c, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class MuLUTConv(nn.Module):
    """ Wrapper of a generalized (spatial-wise) MuLUT block.
        By specifying the unfolding patch size and pixel indices,
        arbitrary sampling pattern can be implemented.
    """

    def __init__(self, mode, nf=64, out_c=None, dense=True, stride=1):
        super(MuLUTConv, self).__init__()
        self.mode = mode

        match mode:
            case 'SxN':
                self.model = MuLUTConvUnit('2x2', nf, out_c=out_c, dense=dense)
                self.K = 2
            case 'DxN':
                self.model = MuLUTConvUnit('2x2d', nf, out_c=out_c, dense=dense)
                self.K = 3
            case 'YxN':
                self.model = MuLUTConvUnit('1x4', nf, out_c=out_c, dense=dense)
                self.K = 3
            case _:
                raise AttributeError("Invalid mode: %s" % mode)

        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.unfold(x, self.K, stride=self.stride)  # B,C*K*K,L
        x = x.reshape(B, C, self.K * self.K,
                      ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), self.K,
                      self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,1,K,K

        if 'Y' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'H' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 2, 3], x[:, :, 3, 2]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        elif 'O' in self.mode:
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2],
                           x[:, :, 1, 3], x[:, :, 3, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)


        x = self.model(x)  # B*C*L,1,1,1 or B*C*L,1,upscale,upscale



        x = x.reshape(B, C, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1), -1)  # B,C,L,out_c
        x = x.permute((0, 1, 3, 2))  # B,C,out_c,L
        x = x.reshape(B, -1, ((H - self.K) // self.stride + 1) * ((W - self.K) // self.stride + 1))  # B,C*out_c,L
        x = F.fold(x, ((H - self.K) // self.stride + 1, (W - self.K) // self.stride + 1), (1, 1), stride=(1, 1))

        return x

class MuLUTcUnit(nn.Module):
    """ Channel-wise MuLUT block [RGB(3D) to RGB(3D)]. """

    def __init__(self, in_c, out_c, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == '1x1':
            self.conv1 = Conv(in_c, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, out_c, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x



if __name__ == '__main__':
    pass
