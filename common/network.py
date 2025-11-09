import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Tuple, List, Optional, final, override
from beartype import beartype
from collections import OrderedDict
from lut_module import (
    ExportableLUTModule,
    iter_input_tensor,
    LUTConfig,
    DFCConfig,
    get_diagonal_input_tensor,
)
from interpolation import DfcArgs, InterpWithVmap
from jaxtyping import Float, jaxtyped
from torch import Tensor


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("total number of parameters: %.3f K" % (num_params / 1e3))


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
    """2D convolution w/ MSRA init."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ActConv(nn.Module):
    """Conv. with activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(ActConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseConv(nn.Module):
    """Dense connected Conv. with activation."""

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out


############### MuLUT Blocks ###############


@beartype
@final
class Residual(ExportableLUTModule):
    def __init__(self, input_shape: tuple[int, int], num_prev: int):
        super().__init__()
        self.shape = input_shape
        self.weights = nn.Parameter(torch.zeros(self.shape))

    def forward(self, x, prev_x):
        assert x.shape[-2:] == self.shape and prev_x.shape[-2:] == self.shape

        weights = torch.clamp(self.weights, 0, 1)
        averaged = weights * prev_x + (1 - weights) * x

        return averaged


class AutoSample(ExportableLUTModule):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_shape = input_size
        self.sampler = nn.Conv2d(1, 4, input_size)
        self.shuffel = nn.PixelShuffle(2)
        self.nw = input_size**2

    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[-2:] == (
            self.input_shape,
            self.input_shape,
        ), f"Unexpected shape: {x.shape}"
        # x = self.sampler(x)
        # logger.debug(self.sampler.weight)
        w = F.softmax(self.sampler.weight.view(-1, self.nw), dim=1).view_as(
            self.sampler.weight
        )
        x = F.conv2d(x, w, bias=self.sampler.bias * 0)
        x = self.shuffel(x)
        return x


@final
class MuLUTConvUnit(ExportableLUTModule):
    """Generalized (spatial-wise)  MuLUT block."""

    def __init__(self, mode: str, nf: int, out_c: int = 1, dense: bool = True):
        super().__init__()

        self.act = nn.ReLU()
        self.out_c: int = out_c

        if mode == "2x2":
            self.conv1 = Conv(1, nf, 2)
        elif mode == "2x2d":
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == "2x2d3":
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == "1x4":
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

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch 1 2 2"]
    ) -> Float[Tensor, "batch {self.out_c} 1 1"]:
        assert self.lut_weight is None and self.lut_config is None, (
            "Use lut_forward for lookup-table based forward"
        )

        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # return x
        return torch.clamp(x, -1, 1)

    @override
    def export_to_lut(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ) -> bool:
        if cfg.dfc:
            ref2index, dfc_input = get_diagonal_input_tensor(
                interval=cfg.dfc.high_precision_interval,
                dimensions=4,
                diagonal_radius=cfg.dfc.diagonal_radius,
            )
            x = dfc_input.reshape(-1, 1, 2, 2)
            output: Float[Tensor, "batch {self.out_c} 1 1"] = self.forward(x)
            output = torch.clamp(output, -1, 1) * 127
            output = torch.round(output).to(torch.int8).cpu()

            destination[prefix + "diagonal_weight"] = output
            destination[prefix + "ref2index"] = ref2index

        all_output: list[Tensor] = []
        device = next(self.parameters()).device
        for input_tensor in iter_input_tensor(cfg.interval, 4, device=device):
            x = input_tensor.reshape(-1, 1, 2, 2)
            output: Float[Tensor, "batch {self.out_c} 1 1"] = self.forward(x)
            output = torch.clamp(output, -1, 1) * 127
            output = output.cpu()
            all_output.append(output)

        result = torch.cat(all_output, dim=0)
        result = torch.round(result).to(torch.int8)
        destination[prefix + "lut_weight"] = result

        return False

    @override
    def load_from_lut(
        self, cfg: LUTConfig, source: OrderedDict[str, torch.Tensor], prefix: str = ""
    ):
        self.lut_weight = source[prefix + "lut_weight"].float()
        self.lut_config = cfg

        if cfg.dfc:
            self.diagonal_weight = source[prefix + "diagonal_weight"].float()
            self.ref2index = source[prefix + "ref2index"]
        return False

    @override
    @jaxtyped(typechecker=beartype)
    def lut_forward(
        self, x: Float[Tensor, "batch 1 2 2"]
    ) -> Float[Tensor, "batch {self.out_c} 1 1"]:
        assert self.lut_weight is not None and self.lut_config

        dfc_args = None
        if self.lut_config.dfc:
            assert self.ref2index is not None and self.diagonal_weight is not None
            dfc_args = DfcArgs(
                high_precision_interval=self.lut_config.dfc.high_precision_interval,
                diagonal_radius=self.lut_config.dfc.diagonal_radius,
                ref2index=self.ref2index,
                diagonal_weights=self.diagonal_weight,
            )

        x = x * 255
        output = InterpWithVmap(
            self.lut_weight,
            upscale=1,
            img_a=x[:, :, 0:1, 0:1],
            img_b=x[:, :, 0:1, 1:2],
            img_c=x[:, :, 1:2, 0:1],
            img_d=x[:, :, 1:2, 1:2],
            interval=self.lut_config.interval,
            out_c=self.out_c,
            dfc=dfc_args,
        )
        return output / 127


class MuLUTConv(ExportableLUTModule):
    """Wrapper of a generalized (spatial-wise) MuLUT block.
    By specifying the unfolding patch size and pixel indices,
    arbitrary sampling pattern can be implemented.
    """

    def __init__(
        self, mode, sample_size, num_prev, nf=64, out_c=1, dense=True, stride=1
    ):
        super(MuLUTConv, self).__init__()
        self.mode = mode
        self.sampler = AutoSample(sample_size)
        self.residual = Residual((2, 2), num_prev)

        self.model = MuLUTConvUnit("2x2", nf, out_c=out_c, dense=dense)
        self.K = sample_size
        self.P = self.K - 1
        self.S = 1  # PixelShuffle is in ConvBlock, we don't upscale here

        self.stride = stride

    def unfold(self, x):
        """
        Do the convolution sampling
        """
        if x is None:
            return x, None
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P), self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        return x, (B, C, H, W)

    def put_back(self, x, ori_shape):
        B, C, H, W = ori_shape
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(
            x, ((H - self.P) * self.S, (W - self.P) * self.S), self.S, stride=self.S
        )
        return x

    def forward(self, x, prev_x: Optional[List[torch.Tensor]] = None):
        assert isinstance(prev_x, list) or prev_x == None
        # Here, prev_x is unfolded multiple times (previously unfolded as x)
        # TODO: Maybe we can do a speedup here
        x, shape = self.unfold(x)
        x = self.sampler(x)

        if prev_x is not None:
            prev_x = [self.unfold(px)[0] for px in prev_x]
            prev_x = [self.sampler(px) for px in prev_x]
            x = self.residual(x, prev_x)

        x = self.model(x)  # B*C*L,K,K
        # logger.debug(f"shape after model: {x.shape}")

        x = self.put_back(x, shape)

        return x


class MuLUTcUnit(nn.Module):
    """Channel-wise MuLUT block [RGB(3D) to RGB(3D)]."""

    def __init__(self, in_c, out_c, mode, nf):
        super(MuLUTcUnit, self).__init__()
        self.act = nn.ReLU()

        if mode == "1x1":
            self.conv1 = Conv(in_c, nf, 1)
        else:
            raise AttributeError

        self.conv2 = DenseConv(nf, nf)
        self.conv3 = DenseConv(nf + nf * 1, nf)
        self.conv4 = DenseConv(nf + nf * 2, nf)
        self.conv5 = DenseConv(nf + nf * 3, nf)
        self.conv6 = Conv(nf * 5, out_c, 1)

    def forward(self, x, prev_x="Unused"):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


if __name__ == "__main__":
    dfc_config = DFCConfig(high_precision_interval=2, diagonal_radius=2)
    lut_cfg = LUTConfig(interval=4, dfc=None)

    def test_module():
        module = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
        state_dict = module.lut_state_dict(cfg=lut_cfg)

        lut_module = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
        lut_module.load_lut_state_dict(lut_cfg, state_dict)

        x = torch.rand((20, 1, 2, 2))
        y1 = module(x)
        y2 = lut_module(x)

        # May not always pass
        assert torch.allclose(y1, y2, atol=1e-2, rtol=1e-2), (
            "This test may not pass every time, you could try again"
        )

        loss = torch.abs(y1 - y2).sum()
        loss.backward()

    def test_nested():
        module = MuLUTConv(mode="unused", sample_size=3, num_prev=1, out_c=1)
        state_dict = module.lut_state_dict(cfg=lut_cfg)

        lut_module = MuLUTConv(mode="unused", sample_size=3, num_prev=1, out_c=1)
        lut_module.load_lut_state_dict(lut_cfg, state_dict)

        x = torch.rand((8, 1, 6, 6))
        y1 = module(x)
        y2 = lut_module(x)

        assert torch.allclose(y1, y2, atol=1e-2, rtol=1e-2)

        loss = torch.abs(y1 - y2).sum()
        loss.backward()

    test_nested()
    # test_module()
