import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, final, override, Iterable
from beartype import beartype
from collections import OrderedDict
from jaxtyping import Float, UInt8, Int8, jaxtyped
import math
from vmap_helper import vmap
from dataclasses import dataclass


@dataclass
class DFCConfig:
    high_precision_interval: int
    diagonal_radius: int


@dataclass
class LUTConfig:
    interval: int
    dfc: DFCConfig | None


class ExportableLUTModule(nn.Module):
    """
    All classes from root model to the LUT-exportable unit should be ExportableLUTModule.
    The rest can be torch.nn.Module.
    For example:

    ```
    A
    |-B
    |-C
      |-D (will export to lut, where you override export_to_lut, load_from_lut and lut_forwar)
    ```
    Then A,C,D should be ExportableLUTModule. B can be `torch.nn.Module`.
    """

    def __init__(self):
        super().__init__()
        self.lut_weight: Tensor | None = None
        self.lut_config: LUTConfig | None = None
        self.diagonal_weight: Tensor | None = None
        self.ref2index: Tensor | None = None

    @override
    def __call__(self, *args, **kwargs):
        if self.lut_weight is not None and self.lut_config is not None:
            return self.lut_forward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)

    def lut_state_dict(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        if destination is None:
            destination = OrderedDict()

        go_down = self.export_to_lut(cfg, destination, prefix, keep_vars)
        if not go_down:
            return destination

        for name, module in self._modules.items():
            new_prefix = prefix + name + "."
            if isinstance(module, ExportableLUTModule):
                _ = module.lut_state_dict(cfg, destination, new_prefix)
            elif isinstance(module, torch.nn.Module):
                _ = module.state_dict(
                    destination=destination, prefix=new_prefix, keep_vars=keep_vars
                )

        return destination

    def load_lut_state_dict(
        self, cfg: LUTConfig, source: OrderedDict[str, torch.Tensor], prefix: str = ""
    ):
        go_down = self.load_from_lut(cfg, source, prefix)
        if not go_down:
            return

        for name, module in self._modules.items():
            new_prefix = prefix + name + "."
            if isinstance(module, ExportableLUTModule):
                module.load_lut_state_dict(cfg, source, new_prefix)
            elif isinstance(module, torch.nn.Module):
                state_dict = {
                    k[len(new_prefix) :]: v
                    for k, v in source.items()
                    if k.startswith(new_prefix)
                }
                _ = module.load_state_dict(state_dict=state_dict)

    def export_to_lut(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ) -> bool:
        """
        Export current module to look-up table.

        If this module contains a submodule (or submodule's submodule) that can be exported
        to LUT, but this isn't, don't override this method.

        Returns whether this module's submodules should be recursively processed. If you override
        this method, you should return False.
        """
        # If this is not overriden, then it means the current module
        # contains an LUT-exporable module, but itself isn't meant to
        # be exported.
        # Therefore, we save only buffers and parameters. Submodules are
        # handled by `lut_state_dict`.
        self._save_to_state_dict(destination, prefix, keep_vars)

        # And continue to submodules
        return True

    def load_from_lut(
        self, cfg: LUTConfig, source: OrderedDict[str, torch.Tensor], prefix: str = ""
    ) -> bool:
        """
        Load current module to look-up table.

        If this module contains a submodule (or submodule's submodule) that can be exported
        to LUT, but this isn't, don't override this method.

        Returns whether this module's submodules should be recursively processed. If you override
        this method, you should return False.
        """
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

        return True

    def lut_forward(self, *args, **kwargs):
        # By default we do the normal foward
        return self.forward(*args, **kwargs)


@jaxtyped(typechecker=beartype)
def get_input_tensor(
    interval: int, dimensions: int
) -> Float[Tensor, "(257//(2**{interval})+1)**{dimensions} {dimensions}"]:
    q = 2**interval
    length = torch.arange(0, 257, q).shape[0]

    @vmap
    def gen_input_part(index):
        result = index.new_zeros((dimensions,))
        for idx in range(dimensions):
            result[dimensions - 1 - idx] = (index // length**idx) % length
        return result

    indicies = torch.arange(length**dimensions)
    enumerated = gen_input_part(indicies) * q
    enumerated[enumerated == 256] = 255
    result = enumerated.float() / 255
    return result


@jaxtyped(typechecker=beartype)
def iter_input_tensor(
    interval: int,
    dimensions: int,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
) -> Iterable[Float[Tensor, "{batch_size} {dimensions}"]]:
    input_tensor = get_input_tensor(interval, dimensions)
    total = input_tensor.shape[0]
    batches = math.ceil(total / batch_size)

    for idx in range(batches):
        batch = input_tensor[idx * batch_size : (idx + 1) * batch_size].to(device)
        yield batch


def get_diagonal_input_tensor(
    interval: int,
    dimensions: int,
    diagonal_radius: int,
) -> tuple[Tensor, Tensor]:
    """
    Returns (ref2index, input_tensor)
    """
    q = 2**interval
    length = torch.arange(0, 257, q).shape[0]
    all_indicies = torch.arange(length**dimensions)
    full_input_tensor = get_input_tensor(interval, dimensions) * 255

    @vmap
    def keep(index):
        in_patch = full_input_tensor[index]

        close = index.new_ones(())
        for dim in range(dimensions):
            close = close.logical_and(
                torch.abs(in_patch[0] - in_patch[dim]) <= diagonal_radius * q
            )

        return close

    keep_mask = keep(all_indicies)
    dfc_input_tensor = full_input_tensor[keep_mask == True]

    elem_count = torch.cumsum(keep_mask, dim=0)

    @vmap
    def ref2idx(index):
        q = 2**interval
        length = torch.arange(0, 257, q).shape[0]

        # We are generating ref2index[index], where index would eventually correspond
        # to [a, b-a, c-a d-a]. Here we figure out a, b-a, c-a, d-a.
        # ref2index is same shape as lut_weight, in inference you look it up like there were no DFC.
        identity = lambda x: x
        # e.g. ref2index[a, -2, ...] we get -2 from length-2
        wrap_around = lambda x: x - length

        ref2idx_query = index.new_zeros((dimensions,))
        for idx in range(dimensions):
            query = (index // length**idx) % length
            query = torch.cond(
                query >= length - diagonal_radius, wrap_around, identity, (query,)
            )
            ref2idx_query[dimensions - 1 - idx] = query

        # Now that we've got a, b-a, c-a, d-a, in ref2idx_query,
        # we figure out abcd and look up it in elem_count to determine our index.
        # We do not use abcd because dimensions might not be 4.
        input_patch = torch.zeros_like(ref2idx_query)
        input_patch[0] = ref2idx_query[0]
        for idx in range(1, dimensions):
            input_patch[idx] = input_patch[0] + ref2idx_query[idx]
        # Now we've got a,b,c,d in input_patch, figure out the index
        elem_count_index = 0
        for idx in range(dimensions):
            elem_count_index += input_patch[dimensions - idx - 1] * length**idx

        self_index = elem_count[elem_count_index] - 1
        return self_index

    ref2index = ref2idx(all_indicies).reshape((length,) * dimensions)

    return ref2index, dfc_input_tensor


if __name__ == "__main__":
    ref2index, dfc_input = get_diagonal_input_tensor(
        interval=4, dimensions=2, diagonal_radius=1
    )
