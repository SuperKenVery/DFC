# ruff: noqa: F722
import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, override

import remote_pdb
import torch
import torch.nn as nn
from accelerate import Accelerator
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .vmap_helper import vmap


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

    This does not always work. Pitfalls:
        - You can only use `submodule(x)` in `forward`, rather than `submodule.forward(x)`.
        - ModuleDict could break this tree!
    """

    redirect_state_dict: LUTConfig | None = None
    """
    Whether calls to `_save_to_state_dict` should be redirected to `export_to_lut`?
    If None, don't redirect. If not None, redirect.
    """

    redirect_load_state_dict: tuple[LUTConfig, Accelerator] | None = None
    """
    Whether to hijack `_load_from_state_dict` to `load_from_lut`?
    """

    def __init__(self):
        super().__init__()
        self.lut_weight: nn.Parameter | None = None
        self.lut_config: LUTConfig | None = None
        self.diagonal_weight: nn.Parameter | None = None
        self.ref2index: Tensor | None = None
        self.export_to_lut_post_hook: Callable[[], None] = lambda: (
            print("Called post hook without calling block_submodule_state_load_save"),
            remote_pdb.set_trace(),
        )[0]
        self.load_from_lut_post_hook: Callable[[], None] = lambda: (
            print("Called post hook without calling block_submodule_state_load_save"),
            remote_pdb.set_trace(),
        )[0]

    def block_submodule_state_load_save(self):
        """
        Block submodules from loading or saving state_dict.

        If this module is meant to be exported into LUT, you should
        call this in `__init__`. It will only do its work under corresponding
        context managers.

        You should also call `self.export_to_lut_post_hook` and `self.load_from_lut_post_hook`
        in appropriate places.
        """
        tmp_modules = {}

        def export_to_lut_post_hook():
            nonlocal tmp_modules
            if ExportableLUTModule.redirect_state_dict:
                tmp_modules = self._modules
                self._modules = {}

        self.export_to_lut_post_hook = export_to_lut_post_hook

        def state_dict_post_hook(self, destination, prefix, local_metadata):
            if ExportableLUTModule.redirect_state_dict:
                self._modules = tmp_modules

        def load_from_lut_post_hook():
            nonlocal tmp_modules
            if ExportableLUTModule.redirect_load_state_dict:
                tmp_modules = self._modules
                self._modules = {}

        self.load_from_lut_post_hook = load_from_lut_post_hook

        def load_state_dict_post_hook(module, incompatible_keys):
            if ExportableLUTModule.redirect_load_state_dict:
                self._modules = tmp_modules

        self.register_state_dict_post_hook(state_dict_post_hook)
        self.register_load_state_dict_post_hook(load_state_dict_post_hook)

    @override
    def __call__(self, *args, **kwargs):
        if self.lut_weight is not None and self.lut_config is not None:
            # print(
            #     f"{self.__class__.__name__}: being called, have weight={self.lut_weight is not None}, have config={self.lut_config is not None} using lut forward"
            # )
            return self.lut_forward(*args, **kwargs)
        else:
            # print(
            #     f"{self.__class__.__name__}: being called, have weight={self.lut_weight is not None}, have config={self.lut_config is not None} using normal forward"
            # )
            return self.forward(*args, **kwargs)

    @staticmethod
    @contextmanager
    def save_as_lut(cfg: LUTConfig):
        """When calling state_dict, call lut_state_dict instead."""
        ExportableLUTModule.redirect_state_dict = cfg
        yield
        ExportableLUTModule.redirect_state_dict = None

    @staticmethod
    @contextmanager
    def load_state_from_lut(cfg: LUTConfig, accelerator: Accelerator):
        """
        Hijack _load_from_state_dict to load_from_lut
        """
        ExportableLUTModule.redirect_load_state_dict = (cfg, accelerator)
        yield
        ExportableLUTModule.redirect_load_state_dict = None

    @override
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.redirect_load_state_dict is None:
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            cfg, accelerator = self.redirect_load_state_dict
            return self.load_from_lut(cfg, accelerator, state_dict, prefix)

    @override
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.redirect_state_dict is None:
            super()._save_to_state_dict(destination, prefix, keep_vars)
        else:
            cfg = self.redirect_state_dict
            self.export_to_lut(cfg, destination, prefix, keep_vars)

    def export_to_lut(
        self,
        cfg: LUTConfig,
        destination: OrderedDict[str, torch.Tensor],
        prefix: str,
        keep_vars: bool,
    ):
        """
        Export current module to look-up table. Corresponds to _save_to_state_dict.

        If this module contains a submodule (or submodule's submodule) that can be exported
        to LUT, but this isn't, don't override this method.

        If you override this method, you need to prevent `state_dict` from recursively calling
        your submodules to save state dict by calling self.block_submodule_state_load_save() in
        __init__ and call self.export_to_lut_post_hook at the end of this method.
        """
        # If this is not overriden, then it means the current module
        # contains an LUT-exporable module, but itself isn't meant to
        # be exported.
        # Therefore, we save only buffers and parameters. Submodules are
        # handled by `lut_state_dict`.
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def load_from_lut(
        self,
        cfg: LUTConfig,
        accelerator: Accelerator,
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str = "",
    ):
        """
        Load current module to look-up table. Corresponds to _load_from_state_dict.

        If this module contains a submodule (or submodule's submodule) that can be exported
        to LUT, but this isn't, don't override this method.

        If you override this method, you need to prevent `state_dict` from recursively calling
        your submodules to save state dict by calling self.block_submodule_state_load_save() in
        __init__ and call self.load_from_lut_post_hook at the end of this method.
        """
        # By default, we load normally
        missing_keys, unexpected_keys, error_msgs = [], [], []
        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata={},
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

        if missing_keys or unexpected_keys or error_msgs:
            raise ValueError(
                f"{self.__class__.__name__} error loading from lut state: err={error_msgs}, missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

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
    batch_size: int = 16,
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
    device: torch.device = torch.device("cuda"),
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
    dfc_input_tensor = full_input_tensor[keep_mask] / 255

    elem_count = torch.cumsum(keep_mask, dim=0)

    @vmap
    def ref2idx(index):
        q = 2**interval
        length = torch.arange(0, 257, q).shape[0]

        # We are generating ref2index[index], where index would eventually correspond
        # to [a, b-a, c-a d-a]. Here we figure out a, b-a, c-a, d-a.
        # ref2index is same shape as lut_weight, in inference you look it up like there were no DFC.
        def identity(x):
            return x

        # e.g. ref2index[a, -2, ...] we get -2 from length-2
        def wrap_around(x):
            return x - length

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

    return ref2index.to(device), dfc_input_tensor.to(device)


if __name__ == "__main__":
    from .network import MuLUTConv

    accelerator = Accelerator()

    dfc_config = DFCConfig(high_precision_interval=4, diagonal_radius=9)
    lut_cfg = LUTConfig(interval=4, dfc=dfc_config)

    def test_save():
        module = MuLUTConv(mode="unused", sample_size=3, num_prev=1, out_c=1)

        with module.save_as_lut(lut_cfg):
            state_dict = module.state_dict()

        lut_module = MuLUTConv(mode="unused", sample_size=3, num_prev=1, out_c=1)
        lut_module = accelerator.prepare(lut_module)
        with lut_module.load_state_from_lut(lut_cfg, accelerator):
            lut_module.load_state_dict(state_dict)

        print(state_dict.keys())
        import pdb

        pdb.set_trace()

    test_save()
