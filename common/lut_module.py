import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, final, override
from beartype import beartype
from collections import OrderedDict
from jaxtyping import Float, UInt8, Int8, jaxtyped
import math
from dataclasses import dataclass


@dataclass
class DFCConfig:
    high_precision_interval: int
    diagonal_width: int


@dataclass
class LUTConfig:
    interval: int
    dfc: DFCConfig | None


class ExportableLUTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lut_weights: Tensor | None = None
        self.lut_config: LUTConfig | None = None

    def lut_state_dict(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        destination = destination or OrderedDict()

        self.export_to_lut(cfg, destination, prefix, keep_vars)

        for name, module in self._modules.items():
            if not isinstance(module, ExportableLUTModule):
                continue
            _ = module.lut_state_dict(cfg, destination, prefix + name + ".")

        return destination

    def load_lut_state_dict(
        self, cfg: LUTConfig, source: OrderedDict[str, torch.Tensor], prefix: str = ""
    ):
        self.load_from_lut(cfg, source, prefix)

        for name, module in self._modules.items():
            if not isinstance(module, ExportableLUTModule):
                continue
            module.load_lut_state_dict(cfg, source, prefix + name + ".")

    def export_to_lut(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ):
        # By default, we save normally
        self._save_to_state_dict(destination, prefix, keep_vars)

    def load_from_lut(
        self, cfg: LUTConfig, source: OrderedDict[str, torch.Tensor], prefix: str = ""
    ):
        # By default, we load normally
        missing_keys, unexpected_keys, error_msgs = [], [], []
        self._load_from_state_dict(
            state_dict=source,
            prefix=prefix,
            local_metadata={},
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

        if missing_keys or unexpected_keys or error_msgs:
            raise ValueError(f"Error loading from lut state: {error_msgs}")

    def lut_forward(self, *args, **kwargs):
        # By default we do the normal foward
        return self.forward(*args, **kwargs)


@jaxtyped(typechecker=beartype)
def get_input_tensor(
    interval: int, dimensions: int
) -> Float[Tensor, "(257//(2**{interval})+1)**{dimensions} 1 2 2"]:
    q = 2**interval
    length = torch.arange(0, 257, q).shape[0]

    @torch.vmap
    def gen_input_part(index):
        result = index.new_zeros((dimensions,))
        for idx in range(dimensions):
            result[dimensions - 1 - idx] = (index // length**idx) % length
        return result

    indicies = torch.arange(length**dimensions)
    enumerated = gen_input_part(indicies) * q
    enumerated[enumerated == 256] = 255
    result = enumerated.float() / 255
    return result.reshape(-1, 1, 2, 2)


def iter_input_tensor(
    interval: int,
    dimensions: int,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
):
    input_tensor = get_input_tensor(interval, dimensions)
    total = input_tensor.shape[0]
    batches = math.ceil(total / batch_size)

    for idx in range(batches):
        batch = input_tensor[idx * batch_size : (idx + 1) * batch_size].to(device)
        yield batch
