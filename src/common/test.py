import random

import torch

from .interpolation import InterpWithVmap


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
            weight, upscale, img_a, img_b, img_c, img_d, interval, out_c, dfc=None, rsc=None
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


def test_rsc_roundtrip():
    """
    Test that RSC export/load/forward roundtrip works, and that
    D4-equivalent inputs produce identical outputs.
    """
    from accelerate import Accelerator

    from .lut_module import LUTConfig, RSCConfig
    from .network import MuLUTConvUnit

    accelerator = Accelerator()
    lut_cfg = LUTConfig(interval=4, dfc=None, rsc=RSCConfig())

    module = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
    module = module.to(accelerator.device)

    # Export
    with module.save_as_lut(lut_cfg):
        state = module.state_dict()

    L = 17
    assert state["lut_weight"].shape[0] < L**4, "RSC should compress the LUT"
    print(f"Compressed {L**4} -> {state['lut_weight'].shape[0]} entries")

    # Load
    lut = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
    lut = accelerator.prepare(lut)
    ulut = accelerator.unwrap_model(lut)
    with ulut.load_state_from_lut(lut_cfg, accelerator):
        ulut.load_state_dict(state)

    # Forward works
    x = torch.rand((16, 1, 2, 2)).to(accelerator.device)
    y = lut(x)
    assert y.shape == (16, 1, 1, 1)

    # Re-export is bit-exact
    with ulut.save_as_lut(lut_cfg):
        state2 = ulut.state_dict()
    assert (state["lut_weight"].cpu() == state2["lut_weight"].cpu()).all()
    assert (state["rot2index"].cpu() == state2["rot2index"].cpu()).all()
    print("Re-export is bit-exact")

    # D4-equivalent inputs produce identical outputs
    a, b, c, d = 0.3, 0.5, 0.7, 0.1
    inputs = [
        torch.tensor([[[[a, b], [c, d]]]]),  # identity
        torch.tensor([[[[c, a], [d, b]]]]),  # rot90
        torch.tensor([[[[d, c], [b, a]]]]),  # rot180
        torch.tensor([[[[b, d], [a, c]]]]),  # rot270
        torch.tensor([[[[b, a], [d, c]]]]),  # hflip
        torch.tensor([[[[c, d], [a, b]]]]),  # vflip
        torch.tensor([[[[a, c], [b, d]]]]),  # diag
        torch.tensor([[[[d, b], [c, a]]]]),  # anti-diag
    ]
    outputs = [lut(inp.to(accelerator.device)).item() for inp in inputs]
    assert all(o == outputs[0] for o in outputs), f"D4 outputs differ: {outputs}"
    print(f"All 8 D4-equivalent inputs produce identical output: {outputs[0]:.6f}")

    # Gradients flow
    x2 = torch.rand((4, 1, 2, 2), device=accelerator.device)
    lut(x2).sum().backward()
    assert ulut.lut_weight.grad is not None
    print("Gradients flow through RSC LUT")

    print("test_rsc_roundtrip passed")
    return True


def test_rsc_with_dfc():
    """Test that RSC + DFC work together."""
    from accelerate import Accelerator

    from .lut_module import DFCConfig, LUTConfig, RSCConfig
    from .network import MuLUTConvUnit

    accelerator = Accelerator()
    dfc_cfg = DFCConfig(high_precision_interval=4, diagonal_radius=2)
    lut_cfg = LUTConfig(interval=5, dfc=dfc_cfg, rsc=RSCConfig())

    module = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
    module = module.to(accelerator.device)

    with module.save_as_lut(lut_cfg):
        state = module.state_dict()

    assert "lut_weight" in state
    assert "rot2index" in state
    assert "diagonal_weight" in state
    assert "ref2index" in state
    print(
        f"RSC+DFC: main LUT {state['lut_weight'].shape[0]} entries, "
        f"diagonal {state['diagonal_weight'].shape[0]} entries"
    )

    lut = MuLUTConvUnit(mode="2x2", nf=64, out_c=1, dense=True)
    lut = accelerator.prepare(lut)
    ulut = accelerator.unwrap_model(lut)
    with ulut.load_state_from_lut(lut_cfg, accelerator):
        ulut.load_state_dict(state)

    x = torch.rand((16, 1, 2, 2)).to(accelerator.device)
    y = lut(x)
    assert y.shape == (16, 1, 1, 1)
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    print("test_rsc_with_dfc passed")
    return True


if __name__ == "__main__":
    test_interpolation()
    test_rsc_roundtrip()
    test_rsc_with_dfc()
