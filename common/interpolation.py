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


# >>> torch._dynamo.list_backends()
# ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']


@jaxtyped(typechecker=typechecker)
# @torch.compile(backend="eager", fullgraph=True)
def InterpWithVmap(
    weight: Float[Tensor, "(2**(8-{interval})+1)**4 {out_c} {upscale} {upscale}"],
    upscale,
    img_a: img_part_type,
    img_b: img_part_type,
    img_c: img_part_type,
    img_d: img_part_type,
    interval: int,
    out_c: int,
    dfc: None | DfcArgs,
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
        Generate the P tensor, for a single group of abcd
        where P 0bijkl means:
            Use abcd to lookup the LUT, but when quantizing,
            mask_bit=1 -> ceil, mask_bit=0 -> floor
            where a's mask bit is i, b's mask bit is j, etc.
        """

        def get_abcd(
            q: int, index: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            img_a_q = torch.floor_divide(img_a, q).long()
            img_b_q = torch.floor_divide(img_b, q).long()
            img_c_q = torch.floor_divide(img_c, q).long()
            img_d_q = torch.floor_divide(img_d, q).long()

            a = img_a_q + (index & 0b1000).bool().long()
            b = img_b_q + (index & 0b0100).bool().long()
            c = img_c_q + (index & 0b0010).bool().long()
            d = img_d_q + (index & 0b0001).bool().long()

            return a, b, c, d

        @vmap
        def _gen_p(
            index: torch.Tensor,
        ) -> Float[Tensor, "{out_c} {upscale} {upscale}"]:
            L = 2 ** (8 - low_prec_interval) + 1

            a, b, c, d = get_abcd(low_prec_q, index)
            idx = a * L**3 + b * L**2 + c * L**1 + d
            return weight[idx]

        @vmap
        def _gen_p_diagonal(
            index: torch.Tensor,
        ) -> Float[Tensor, "{out_c} {upscale} {upscale}"]:
            """
            Generates P for abcd near the diagonal.

            When the pixels are not really near the diagonal, this function outputs useless data. But it
            doesn't fail.
            """
            assert dfc is not None

            a, b, c, d = get_abcd(high_prec_q, index)

            idx = dfc.ref2index[
                a,
                torch.clamp(b - a, -dfc.diagonal_radius, dfc.diagonal_radius),
                torch.clamp(c - a, -dfc.diagonal_radius, dfc.diagonal_radius),
                torch.clamp(d - a, -dfc.diagonal_radius, dfc.diagonal_radius),
            ]
            return dfc.diagonal_weights[idx]

        indicies = torch.arange(16, dtype=torch.int, device=img_a.device)

        if along_diagonal:
            assert dfc is not None
            return _gen_p_diagonal(indicies)
        else:
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

        1. Sort abcd, into xyzt where x>=y>=z>=t
        2. w0, w1, ..., w4 = W-x, x-y, y-z, z-t, t
        3. O0 = P0000, O4 = P1111
        4. O1, O2, O3 is a little bit complex.
            1) a->0b1000 b->0b0100
               c->0b0010 d->0b0001
            2) Sort the above bitmask with abcd,
               so that xyzt each also have a bitmask
            3) O1=P x.mask, O2=P x.mask|y.mask, O3=P x.mask|y.mask|z.mask
               where P 0bijkl means:
                Use abcd to lookup the LUT, but when quantizing,
                mask_bit=1 -> ceil, mask_bit=0 -> floor
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

        values = torch.stack([mod_a, mod_b, mod_c, mod_d])  # shape: (4,)

        # Stable sort descending: larger values first, equal values maintain original index order
        # argsort with stable=True ensures that when values are equal, original order is preserved
        sorted_positions = torch.argsort(values, descending=True, stable=True)

        # Map back to the original values and masks
        xyzt_values = torch.cat([
            torch.tensor(q, dtype=torch.float, device=img_a.device).unsqueeze(0),
            values[sorted_positions],
            torch.tensor(0, dtype=torch.float, device=img_a.device).unsqueeze(0)
        ])

        masks = torch.tensor(
            [0b1000, 0b0100, 0b0010, 0b0001],
            dtype=torch.long,
            device=img_a.device,
        )
        xyzt_indicies = torch.cat([
            torch.tensor([0b0000], dtype=torch.long, device=img_a.device),
            masks[sorted_positions],
            torch.tensor([0b0000], dtype=torch.long, device=img_a.device)
        ])

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


def _reference_interp_single(weight, a, b, c, d, interval, out_c, upscale):
    """
    Reference implementation for a single pixel group (a,b,c,d).
    This directly implements the 24-case tetrahedral interpolation from model.py.

    a, b, c, d: scalar float values in [0, 255]
    Returns: tensor of shape (out_c, upscale, upscale)
    """
    q = 2**interval
    L = 2 ** (8 - interval) + 1

    # Quantize
    a1 = int(a // q)
    b1 = int(b // q)
    c1 = int(c // q)
    d1 = int(d // q)

    a2 = a1 + 1
    b2 = b1 + 1
    c2 = c1 + 1
    d2 = d1 + 1

    # Fractional parts
    fa = a % q
    fb = b % q
    fc = c % q
    fd = d % q

    # Get all 16 corner values
    def get_p(ai, bi, ci, di):
        idx = ai * L**3 + bi * L**2 + ci * L + di
        return weight[idx].flatten()  # (out_c * upscale * upscale,)

    p0000 = get_p(a1, b1, c1, d1)
    p0001 = get_p(a1, b1, c1, d2)
    p0010 = get_p(a1, b1, c2, d1)
    p0011 = get_p(a1, b1, c2, d2)
    p0100 = get_p(a1, b2, c1, d1)
    p0101 = get_p(a1, b2, c1, d2)
    p0110 = get_p(a1, b2, c2, d1)
    p0111 = get_p(a1, b2, c2, d2)
    p1000 = get_p(a2, b1, c1, d1)
    p1001 = get_p(a2, b1, c1, d2)
    p1010 = get_p(a2, b1, c2, d1)
    p1011 = get_p(a2, b1, c2, d2)
    p1100 = get_p(a2, b2, c1, d1)
    p1101 = get_p(a2, b2, c1, d2)
    p1110 = get_p(a2, b2, c2, d1)
    p1111 = get_p(a2, b2, c2, d2)

    # Determine the simplex (24 cases based on ordering of fa, fb, fc, fd)
    fab = fa > fb
    fac = fa > fc
    fad = fa > fd
    fbc = fb > fc
    fbd = fb > fd
    fcd = fc > fd

    # Case 1: a >= b >= c >= d
    if fab and fbc and fcd:
        out = (
            (q - fa) * p0000
            + (fa - fb) * p1000
            + (fb - fc) * p1100
            + (fc - fd) * p1110
            + fd * p1111
        )
    # Case 2: a >= b >= d > c
    elif fab and fbc and fbd and not fcd:
        out = (
            (q - fa) * p0000
            + (fa - fb) * p1000
            + (fb - fd) * p1100
            + (fd - fc) * p1101
            + fc * p1111
        )
    # Case 3: a >= d > b >= c
    elif fab and fbc and fad and not fbd:
        out = (
            (q - fa) * p0000
            + (fa - fd) * p1000
            + (fd - fb) * p1001
            + (fb - fc) * p1101
            + fc * p1111
        )
    # Case 4: d > a >= b >= c
    elif fab and fbc and not fad:
        out = (
            (q - fd) * p0000
            + (fd - fa) * p0001
            + (fa - fb) * p1001
            + (fb - fc) * p1101
            + fc * p1111
        )
    # Case 5: a >= c > b, a >= b >= d (i.e., a >= c > b >= d)
    elif not fbc and fab and fac and fbd:
        out = (
            (q - fa) * p0000
            + (fa - fc) * p1000
            + (fc - fb) * p1010
            + (fb - fd) * p1110
            + fd * p1111
        )
    # Case 6: a >= c >= d > b
    elif not fbc and fab and fac and fcd and not fbd:
        out = (
            (q - fa) * p0000
            + (fa - fc) * p1000
            + (fc - fd) * p1010
            + (fd - fb) * p1011
            + fb * p1111
        )
    # Case 7: a >= d > c > b
    elif not fbc and fab and fac and fad and not fcd:
        out = (
            (q - fa) * p0000
            + (fa - fd) * p1000
            + (fd - fc) * p1001
            + (fc - fb) * p1011
            + fb * p1111
        )
    # Case 8: d > a >= c > b
    elif not fbc and fab and fac and not fad:
        out = (
            (q - fd) * p0000
            + (fd - fa) * p0001
            + (fa - fc) * p1001
            + (fc - fb) * p1011
            + fb * p1111
        )
    # Case 9: c > a >= b >= d
    elif not fbc and not fac and fab and fbd:
        out = (
            (q - fc) * p0000
            + (fc - fa) * p0010
            + (fa - fb) * p1010
            + (fb - fd) * p1110
            + fd * p1111
        )
    # Case 10: c > a >= d > b
    elif not fbc and not fac and fab and fad and not fbd:
        out = (
            (q - fc) * p0000
            + (fc - fa) * p0010
            + (fa - fd) * p1010
            + (fd - fb) * p1011
            + fb * p1111
        )
    # Case 11: c >= d > a >= b
    elif not fbc and not fac and fab and fcd and not fad:
        out = (
            (q - fc) * p0000
            + (fc - fd) * p0010
            + (fd - fa) * p0011
            + (fa - fb) * p1011
            + fb * p1111
        )
    # Case 12: d > c > a >= b
    elif not fbc and not fac and fab and not fcd:
        out = (
            (q - fd) * p0000
            + (fd - fc) * p0001
            + (fc - fa) * p0011
            + (fa - fb) * p1011
            + fb * p1111
        )
    # Case 13: b > a >= c >= d
    elif not fab and fac and fcd:
        out = (
            (q - fb) * p0000
            + (fb - fa) * p0100
            + (fa - fc) * p1100
            + (fc - fd) * p1110
            + fd * p1111
        )
    # Case 14: b > a >= d > c
    elif not fab and fac and fad and not fcd:
        out = (
            (q - fb) * p0000
            + (fb - fa) * p0100
            + (fa - fd) * p1100
            + (fd - fc) * p1101
            + fc * p1111
        )
    # Case 15: b >= d > a >= c
    elif not fab and fac and fbd and not fad:
        out = (
            (q - fb) * p0000
            + (fb - fd) * p0100
            + (fd - fa) * p0101
            + (fa - fc) * p1101
            + fc * p1111
        )
    # Case 16: d > b > a >= c
    elif not fab and fac and not fbd:
        out = (
            (q - fd) * p0000
            + (fd - fb) * p0001
            + (fb - fa) * p0101
            + (fa - fc) * p1101
            + fc * p1111
        )
    # Case 17: b >= c > a >= d
    elif not fab and not fac and fbc and fad:
        out = (
            (q - fb) * p0000
            + (fb - fc) * p0100
            + (fc - fa) * p0110
            + (fa - fd) * p1110
            + fd * p1111
        )
    # Case 18: b >= c >= d > a
    elif not fab and not fac and fbc and fcd and not fad:
        out = (
            (q - fb) * p0000
            + (fb - fc) * p0100
            + (fc - fd) * p0110
            + (fd - fa) * p0111
            + fa * p1111
        )
    # Case 19: b >= d > c > a
    elif not fab and not fac and fbc and fbd and not fcd:
        out = (
            (q - fb) * p0000
            + (fb - fd) * p0100
            + (fd - fc) * p0101
            + (fc - fa) * p0111
            + fa * p1111
        )
    # Case 20: d > b >= c > a
    elif not fab and not fac and fbc and not fbd:
        out = (
            (q - fd) * p0000
            + (fd - fb) * p0001
            + (fb - fc) * p0101
            + (fc - fa) * p0111
            + fa * p1111
        )
    # Case 21: c > b > a >= d
    elif not fab and not fac and not fbc and fad:
        out = (
            (q - fc) * p0000
            + (fc - fb) * p0010
            + (fb - fa) * p0110
            + (fa - fd) * p1110
            + fd * p1111
        )
    # Case 22: c > b >= d > a
    elif not fab and not fac and not fbc and fbd and not fad:
        out = (
            (q - fc) * p0000
            + (fc - fb) * p0010
            + (fb - fd) * p0110
            + (fd - fa) * p0111
            + fa * p1111
        )
    # Case 23: c >= d > b > a
    elif not fab and not fac and not fbc and fcd and not fbd:
        out = (
            (q - fc) * p0000
            + (fc - fd) * p0010
            + (fd - fb) * p0011
            + (fb - fa) * p0111
            + fa * p1111
        )
    # Case 24: d > c > b > a
    else:
        out = (
            (q - fd) * p0000
            + (fd - fc) * p0001
            + (fc - fb) * p0011
            + (fb - fa) * p0111
            + fa * p1111
        )

    return (out / q).reshape(out_c, upscale, upscale)


def test_interpolation():
    """
    Test InterpWithVmap against the reference implementation.
    """
    import random

    # Test parameters
    interval = 4
    q = 2**interval
    L = 2 ** (8 - interval) + 1
    out_c = 2
    upscale = 1

    # Create a random LUT weight
    weight = torch.randn(L**4, out_c, upscale, upscale)

    # Test multiple random inputs
    errors = []
    for _ in range(100):
        # Random pixel values
        a = random.uniform(0, 255)
        b = random.uniform(0, 255)
        c = random.uniform(0, 255)
        d = random.uniform(0, 255)

        # Reference result
        ref_result = _reference_interp_single(
            weight, a, b, c, d, interval, out_c, upscale
        )

        # InterpWithVmap result
        img_a = torch.tensor([[[[a]]]])
        img_b = torch.tensor([[[[b]]]])
        img_c = torch.tensor([[[[c]]]])
        img_d = torch.tensor([[[[d]]]])

        vmap_result = InterpWithVmap(
            weight, upscale, img_a, img_b, img_c, img_d, interval, out_c, dfc=None
        )
        # vmap_result shape: (B=1, C*out_c=2, ch=1, cw=1)
        # Reshape to match reference: (out_c, upscale, upscale)
        vmap_result = vmap_result[0].reshape(out_c, upscale, upscale)

        error = torch.abs(ref_result - vmap_result).max().item()
        errors.append(error)

        if error > 1e-5:
            print(f"MISMATCH: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
            print(f"  fa={a % q:.2f}, fb={b % q:.2f}, fc={c % q:.2f}, fd={d % q:.2f}")
            print(f"  Reference: {ref_result.flatten()}")
            print(f"  Vmap:      {vmap_result.flatten()}")
            print(f"  Error: {error}")

    print(
        f"\nTest complete. Max error: {max(errors):.6f}, Mean error: {sum(errors) / len(errors):.6f}"
    )
    return max(errors) < 1e-5


if __name__ == "__main__":
    test_interpolation()
