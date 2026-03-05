# DFC max/min diagonal

We are developing RSC (/data/xyh/DFCs/DFC/TASK.md).

Currently, DFC decides whether an input is "along diagnal" by:

- |a-b|<threshold
- |a-c|<threshold
- |a-d|<threshold

Which is

- Not symmetric
- Incompatible with RSC

We should change it to `max(a,b,c,d) - min(a,b,c,d) <= threshold`. This would make it compatible with RSC. So:

1. Implement minmax DFC
2. Implement RSC on top of DFC.


---

Below is some previous conversation, as a reference.

❯ `max(a,b,c,d) - min(a,b,c,d) ≤ T` sounds great! It seems that DFC should have designed like this from the ground up! Allowing b-c to overflow T doesn't seem to make
  sense right?

● Right. With the original design (reference = a), a point like a=100, b=109, c=91, d=100 with dw=9 passes the check (|b-a|=9, |c-a|=9, |d-a|=0, all ≤ 9) and gets
  high-precision treatment. But |b-c|=18 — those two pixels aren't similar at all. That's arguably a waste of high-precision table space on a case that isn't truly "near the
  diagonal."

  max-min ≤ T is a stricter and more uniform criterion. It also happens to give a smaller table ((T+1)³ vs (2T+1)³) while arguably capturing the points that actually matter —
   the ones where all pixels are genuinely close to each other.

  That said, it's possible the original authors chose the positional reference deliberately because the SR-LUT model already rotates the image 4 ways and averages. Each
  rotation makes a different pixel the positional a, so the "overflow" cases for one rotation might be well-covered by another rotation's high-precision region. With RSC
  replacing that rotation-averaging approach, the positional reference loses that justification entirely, making max-min ≤ T the natural choice.

❯ Oh, speaking of that. SR-LUT's rotation actually improves performance a bit. By introducing RSC, will we eliminate that perf boost?

● The rotation-averaging boost comes from two things:

  1. Symmetry enforcement: a horizontal edge and a vertical edge get the same quality treatment, because both orientations are processed and averaged.
  2. Interpolation noise reduction: each rotation hits different quantization grid points, so averaging 4 interpolation paths reduces quantization artifacts (like an
  ensemble).

  RSC fully replaces (1) — it's a hard guarantee rather than a soft average. A LUT with RSC cannot treat a horizontal edge differently from a vertical edge, by construction.

  RSC doesn't directly replace (2). With RSC, all 4 rotations look up the same canonical entry and (if fractional parts sort the same way) produce identical interpolation
  results. So rotating and averaging becomes mostly redundant — you'd get nearly the same output 4 times.

  But there's a compensating factor: RSC gives ~8× compression. You could spend that budget on higher quantization precision (smaller interval → larger L), which directly
  reduces the interpolation error that rotation-averaging was smoothing over. Attacking the root cause rather than averaging over the symptoms.

  In short: yes, RSC eliminates the symmetry benefit of rotation-averaging. But that's the point — it replaces a soft approximation with a hard constraint plus room for
  better precision. If you still want the ensemble effect on top of RSC, you could keep rotation-averaging at inference (costs 4× compute, no extra storage), but it would
  likely be diminishing returns.
