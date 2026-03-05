from collections import namedtuple
from dataclasses import dataclass

import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int64, jaxtyped
from torch import Tensor

from .vmap_helper import vmap

# image_part_type: You do a 2x2 convolution over the original image.
# For the top-left pixel of the conv kernel, what is sees is an image part.
#
# ch and cw means conved width and conved height, this is (w-1) and (h-1) in the simplest case.
#
# When actually calling this function, we often use the shape (batch*channel*L, 1, 2, 2) where L is the total number of sliding window when you conv over the original image.
img_part_type = Float[Tensor, "batch channel ch cw"]


@dataclass
class DfcArgs:
    high_precision_interval: int
    """The higher-precision interval, for data along the diagonal."""

    diagonal_radius: int
    """The horizontal range of high-precision sampling. This is the --dw in original DFC implementation."""

    ref2index: torch.Tensor
    diagonal_weights: torch.Tensor


@dataclass
class RscArgs:
    rot2index: torch.Tensor
    """Flat 1D tensor of shape (L**4,). Maps flat LUT index to RSC-compressed index."""


# >>> torch._dynamo.list_backends()
# ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']


@jaxtyped(typechecker=typechecker)
# @torch.compile(backend="eager", fullgraph=True)
def InterpWithVmap(
    weight: Float[Tensor, "entries {out_c} {upscale} {upscale}"],
    upscale,
    img_a: img_part_type,
    img_b: img_part_type,
    img_c: img_part_type,
    img_d: img_part_type,
    interval: int,
    out_c: int,
    dfc: None | DfcArgs,
    rsc: None | RscArgs,
) -> Float[Tensor, "batch channel*{out_c} ch*{upscale} cw*{upscale}"]:
    """
    Interpolate with vmap.

    img_abcd: 0<=x<=255

    """
    if dfc is None:
        low_prec_interval = high_prec_interval = interval
        low_prec_q = high_prec_q = 2**interval
    else:
        low_prec_interval = interval
        high_prec_interval = dfc.high_precision_interval

        high_prec_q = 2**high_prec_interval
        low_prec_q = 2**low_prec_interval

    def gen_P(
        img_a: Float[Tensor, ""],
        img_b: Float[Tensor, ""],
        img_c: Float[Tensor, ""],
        img_d: Float[Tensor, ""],
        along_diagonal: bool,
    ) -> Float[Tensor, "16 {upscale} {upscale}"]:
        """
        This function takes values and look it up in LUT.

        Generate the P tensor, for a single group of abcd
        where P 0bijkl means:
            Use abcd to lookup the LUT, but when quantizing,
            mask_bit=1 -> ceil, mask_bit=0 -> floor
            where a's mask bit is i, b's mask bit is j, etc.
        """

        def get_abcd(
            q: int, index: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Get the abcd values used to look up LUT.

            The looked up value would be stored in P[index].
            """
            img_a_q = torch.floor_divide(img_a, q).long()
            img_b_q = torch.floor_divide(img_b, q).long()
            img_c_q = torch.floor_divide(img_c, q).long()
            img_d_q = torch.floor_divide(img_d, q).long()

            # At the right index, we ceil rather than floor
            a = img_a_q + (index & 0b1000).bool().long()
            b = img_b_q + (index & 0b0100).bool().long()
            c = img_c_q + (index & 0b0010).bool().long()
            d = img_d_q + (index & 0b0001).bool().long()

            return a, b, c, d

        @vmap
        def _gen_p(
            index: torch.Tensor,
        ) -> Float[Tensor, "{out_c} {upscale} {upscale}"]:
            if along_diagonal:
                assert dfc is not None
                a, b, c, d = get_abcd(high_prec_q, index)
                idx = dfc.ref2index[
                    a,
                    torch.clamp(b - a, -dfc.diagonal_radius, dfc.diagonal_radius),
                    torch.clamp(c - a, -dfc.diagonal_radius, dfc.diagonal_radius),
                    torch.clamp(d - a, -dfc.diagonal_radius, dfc.diagonal_radius),
                ]
                # RSC not applied to diagonal weights: D4 permutations can
                # push near-diagonal points outside the diagonal region.
                return dfc.diagonal_weights[idx]
            else:
                L = 2 ** (8 - low_prec_interval) + 1
                a, b, c, d = get_abcd(low_prec_q, index)
                idx = a * L**3 + b * L**2 + c * L + d

                if rsc is not None:
                    idx = rsc.rot2index[idx]

                return weight[idx]

        indicies = torch.arange(16, dtype=torch.int, device=img_a.device)
        return _gen_p(indicies)

    @vmap
    @vmap
    @vmap
    @vmap
    def _interpolate_dispatcher(
        img_a: Float[Tensor, ""],
        img_b: Float[Tensor, ""],
        img_c: Float[Tensor, ""],
        img_d: Float[Tensor, ""],
    ) -> Float[Tensor, "{out_c} {upscale} {upscale}"]:
        if not dfc:
            return _interpolate(img_a, img_b, img_c, img_d, along_diagonal=False)

        def interp_away_from_diagonal():
            return _interpolate(img_a, img_b, img_c, img_d, False)

        def interp_along_diagonal():
            return _interpolate(img_a, img_b, img_c, img_d, True)

        b_close = torch.abs(img_b - img_a) <= high_prec_q * dfc.diagonal_radius
        c_close = torch.abs(img_c - img_a) <= high_prec_q * dfc.diagonal_radius
        d_close = torch.abs(img_d - img_a) <= high_prec_q * dfc.diagonal_radius
        close = b_close.logical_and(c_close).logical_and(d_close)
        # print(f"Close mask: {close[None]}")

        # return torch.cond(close, interp_along_diagonal, interp_away_from_diagonal)
        return torch.where(close, interp_along_diagonal(), interp_away_from_diagonal())

    def _interpolate(
        img_a: Float[Tensor, ""],
        img_b: Float[Tensor, ""],
        img_c: Float[Tensor, ""],
        img_d: Float[Tensor, ""],
        along_diagonal: bool,
    ) -> Float[Tensor, "{out_c} {upscale} {upscale}"]:
        """
        Tetrahedral interpolation equivalent for 4D space

        If we define
        - The points where LUT has stored value as "precise point"

        Then, tetrahedral interpolation is a weighed average where
        - weights: basically the distance to precise points
        - value: the LUT value at precise points before or after current value.
          We have 4 "current value" (call it a,b,c,d) and we have 5 values to be averaged:
              1. All four values floor to precise point
              2. The largest one ceil to precise point, the rest floor
              3. The largest and second largest ceil, the rest floor
              ...
              5. All ceil to precise point

        1. Sort abcd, into xyzt where x>=y>=z>=t
        2. w0, w1, ..., w4 = W-x, x-y, y-z, z-t, t (these are the weights, W is quantize interval)
        3. O0 = P0000, O4 = P1111 (these are LUT values at different precise points)
            where P 0bijkl means:
            Use abcd to lookup the LUT, but when quantizing,
            mask_bit=1 -> ceil, mask_bit=0 -> floor
        4. O1, O2, O3 is a little bit complex.
            1) a->0b1000 b->0b0100
               c->0b0010 d->0b0001
            2) Sort the above bitmask with abcd,
               so that xyzt each also have a bitmask
            3) O1=P x.mask, O2=P x.mask|y.mask, O3=P x.mask|y.mask|z.mask

        5. sum w_i * O_i, i=0~4
        """
        if along_diagonal:
            q = high_prec_q
        else:
            q = low_prec_q

        # 1. Sort abcd with stable tie-breaking
        # The reference uses: fab = fa > fb, fac = fa > fc, etc.
        # For consistency, when values are equal, maintain original index order: a < b < c < d
        mod_a = img_a % q
        mod_b = img_b % q
        mod_c = img_c % q
        mod_d = img_d % q

        abcd_values = torch.stack(
            [
                torch.tensor(q, dtype=torch.float, device=img_a.device),
                mod_a,
                mod_b,
                mod_c,
                mod_d,
                torch.tensor(0, dtype=torch.float, device=img_a.device),
            ]
        )  # q, a, b, c, d, 0
        abcd_indicies = torch.tensor(
            [0b0000, 0b1000, 0b0100, 0b0010, 0b0001, 0b0000],
            dtype=torch.long,
            device=img_a.device,
        )  # a->0b1000, b->0b0100, c->0b0010, d->0b0001

        # Sort by img_abcd, bit mask still corresponds to its own one
        sorted_indicies = torch.argsort(abcd_values, descending=True, stable=True)
        xyzt_values = abcd_values[sorted_indicies]
        xyzt_indicies = abcd_indicies[sorted_indicies]
        # assert xyzt_values[0] == float(q) and xyzt_indicies[0] == float(0b0000)
        # assert xyzt_values[-1] == float(0) and xyzt_indicies[-1] == float(0b0000)

        # 2. w
        w: Float[Tensor, "5"] = xyzt_values[:-1] - xyzt_values[1:]

        # 3. 4. O
        Oidx = xyzt_indicies.new_zeros(5, dtype=torch.long, device=img_a.device)
        for idx in range(1, 5):
            Oidx[idx] = Oidx[idx - 1].bitwise_or(xyzt_indicies[idx])

        P = gen_P(img_a, img_b, img_c, img_d, along_diagonal)
        O: Float[Tensor, "5 {out_c} {upscale} {upscale}"] = P[Oidx]

        # 5. sum(w * O)
        # Use einsum because O.shape==(5, out_c, upscale, upscale) and upscale can !=1
        # (5,) * (5, 4, 2, 2) isn't possible, need einsum to mean this multiply-and-sum.
        return torch.einsum("i,icjk->cjk", w, O) / q

    interpolated: Float[Tensor, "batch channel ch cw {out_c} {upscale} {upscale}"] = (
        _interpolate_dispatcher(img_a, img_b, img_c, img_d)
    )

    B, C, ch, cw = img_a.shape
    result = interpolated.permute(0, 1, 4, 2, 5, 3, 6).reshape(
        B, C * out_c, ch * upscale, cw * upscale
    )
    return result

