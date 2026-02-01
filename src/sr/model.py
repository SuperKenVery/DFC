import os
import sys
from typing import List, Optional, Tuple, final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from torch import Tensor

from ..common.network import *


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


def identity(input):
    return input


@beartype
@final
class ConvBlock(ExportableLUTModule):
    def __init__(
        self,
        in_c,
        out_c,
        sample_size,
        scale=None,
        output_quant=False,
        branches=3,
        nf=64,
    ):
        super(ConvBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.branches = branches
        self.module_dict = dict()
        self.upscale = scale
        self.output_quant = output_quant
        self.sample_size = sample_size

        scale_factor = 1 if scale is None else scale**2
        for c in range(in_c):
            for b in range(branches):
                key = "DepthwiseBlock{}_{}".format(c, b)
                submodule = MuLUTConv(
                    "Sx{}".format("N"),
                    nf=nf,
                    num_prev=1,
                    sample_size=sample_size,
                    out_c=out_c * scale_factor,
                    stride=1,
                )
                setattr(self, key, submodule)
        if scale is None:
            self.pixel_shuffle = identity
        else:
            self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(
        self,
        x,
        prev_x,
        debug_info: Tuple[str, dict] = ("", {}),
    ):
        prefix, debug_dict = debug_info

        x_out = 0
        for c in range(self.in_c):
            x_c = x[:, c : c + 1, :, :]
            prevx_c = prev_x[:, c : c + 1, :, :] if prev_x != None else None
            debug_dict[f"{prefix}.in_c{c}.input.x"] = x_c
            debug_dict[f"{prefix}.in_c{c}.input.prev_x"] = prevx_c
            pred = 0
            for b in range(self.branches):
                pad = self.sample_size - 1
                key = "DepthwiseBlock{}_{}".format(c, b)
                sub_module = getattr(self, key)
                for r in [0, 1, 2, 3]:
                    y = round_func(
                        torch.tanh(
                            torch.rot90(
                                self.pixel_shuffle(
                                    sub_module(
                                        F.pad(
                                            torch.rot90(x_c, r, [2, 3]),
                                            (0, pad, 0, pad),
                                            mode="replicate",
                                        ),
                                        F.pad(
                                            torch.rot90(prevx_c, r, [2, 3]),
                                            (0, pad, 0, pad),
                                            mode="replicate",
                                        )
                                        if prevx_c != None
                                        else None,
                                        debug_info=(
                                            f"{prefix}.in_c{c}.b{b}.rot{r}",
                                            debug_dict,
                                        ),
                                    )
                                ),
                                (4 - r) % 4,
                                [2, 3],
                            )
                        )
                        * 127
                    )

                    assert (
                        isinstance(pred, torch.Tensor) == False or pred.shape == y.shape
                    ), f"Unexpected shape: pred={pred.shape}, y={y.shape}"
                    pred += y
                    debug_dict[f"{prefix}.in_c{c}.b{b}.rot{r}"] = y / 127

            x_out += pred
            debug_dict[f"{prefix}.in_c{c}"] = pred / (self.branches * 4 * 127)
        if self.output_quant:
            avg_factor = self.branches * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            x = x_out / self.in_c

        return x


class SPF_LUT_net(ExportableLUTModule):
    def __init__(self, sample_size, nf=32, scale=4, branches=3, stages=2):
        super(SPF_LUT_net, self).__init__()
        self.upscale = scale
        self.branches = branches

        conv_args = {
            "in_c": 1,
            "out_c": 2,
            "scale": None,
            "output_quant": False,
            "branches": branches,
            "nf": nf,
            "sample_size": sample_size,
        }

        self.convblock1 = ConvBlock(**conv_args)
        self.convblock2 = ConvBlock(**conv_args)
        self.convblock3 = ConvBlock(**conv_args)
        self.convblock4 = ConvBlock(**{**conv_args, "out_c": 1})
        self.ChannelConv = MuLUTcUnit(in_c=4, out_c=4, mode="1x1", nf=nf)
        self.upblock = ConvBlock(**{**conv_args, "in_c": 4, "out_c": 1, "scale": scale})

    def forward(
        self,
        x,
        phase="train",
        debug_info: Tuple[str, dict] = ("", {}),
    ) -> Tensor:
        prefix, debug_dict = debug_info
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        def _sub_debug_info(name: str) -> Tuple[str, dict]:
            return (f"{prefix}.{name}", debug_dict)

        # block1
        x1 = x
        x = self.convblock1(x, None, debug_info=_sub_debug_info("convblock1"))
        avg_factor, bias, norm = self.branches * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        debug_dict[f"{prefix}.block1_out"] = x

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block2
        x2 = x
        x = self.convblock2(x, x1, debug_info=_sub_debug_info("convblock2"))
        avg_factor, bias, norm = self.branches * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        debug_dict[f"{prefix}.block2_out"] = x

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block3
        x3 = x
        x = self.convblock3(x, x2, debug_info=_sub_debug_info("convblock3"))
        avg_factor, bias, norm = self.branches * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        debug_dict[f"{prefix}.block3_out"] = x

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block4
        x4 = x
        x = self.convblock4(x, x3, debug_info=_sub_debug_info("convblock4"))
        avg_factor, bias, norm = self.branches * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        debug_dict[f"{prefix}.block4_out"] = x
        refine_list.append(x)

        # concat
        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0
        debug_dict[f"{prefix}.channel_conv_out"] = x

        # upblock
        x = self.upblock(
            x, torch.cat([x1, x2, x3, x4], dim=1), debug_info=_sub_debug_info("upblock")
        )
        avg_factor, bias, norm = self.branches, 0, 1
        x = round_func((x / avg_factor) + bias)
        debug_dict[f"{prefix}.upblock_out"] = x / 255.0

        if phase == "train":
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H, self.upscale * W))

        return x
