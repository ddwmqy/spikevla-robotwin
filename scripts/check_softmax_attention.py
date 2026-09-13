"""Invariants for the spike cross-attention weighting modes.

Covers the delicate parts of the "softmax" variant (cross_attention_softmax=softmax):
padded sources must receive no attention mass, rows with no valid source must come out as
exact zeros rather than NaN, weights must form a distribution over the valid sources, and the
legacy "none" mode must keep working unchanged. Runs on CPU.
"""
import math
import sys

root = "/data/260010028/dwh_vla/v4_code"
sys.path.insert(0, root)

import torch  # noqa: E402

from turbovla.models.components.spike_fusion import SpikeBiMultiHeadAttention  # noqa: E402


def make(mode: str, heads: int = 2, dim: int = 8) -> SpikeBiMultiHeadAttention:
    torch.manual_seed(0)
    return SpikeBiMultiHeadAttention(v_dim=dim, l_dim=dim, embed_dim=dim, num_heads=heads, dropout=0.0,
                                     attention_backend="manual", timesteps=2, gradient_checkpointing=False,
                                     softmax_mode=mode)


def weights_of(attn, counts, valid_source):
    return attn._softmax_weights(counts, valid_source)


def main() -> None:
    b, h, nq, ns = 2, 2, 4, 5
    counts = torch.randint(0, 9, (b, h, nq, ns), dtype=torch.float32)

    # 1. weights form a distribution over valid sources only
    valid = torch.ones(b, ns, 1, dtype=torch.bool)
    valid[1, 3:, 0] = False                      # sample 1: last two sources are padding
    w = weights_of(make("softmax"), counts, valid)
    assert torch.isfinite(w).all(), "non-finite weights"
    sums = w.sum(-1)
    assert torch.allclose(sums[0], torch.ones_like(sums[0]), atol=1e-6), f"row sums != 1: {sums[0]}"
    assert torch.allclose(sums[1], torch.ones_like(sums[1]), atol=1e-6), f"row sums != 1: {sums[1]}"
    assert (w[1, :, :, 3:] == 0).all(), "attention mass leaked into padded sources"
    print("1. normalisation + padding exclusion OK")

    # 2. a source set that is entirely padding yields zero weights, not NaN
    none_valid = torch.zeros(b, ns, 1, dtype=torch.bool)
    w_none = weights_of(make("softmax"), counts, none_valid)
    assert torch.isfinite(w_none).all() and (w_none == 0).all(), "all-padding row is not exactly zero"
    print("2. all-padding sources -> exact zeros, no NaN OK")

    # 3. single valid source -> weight 1 regardless of the score scale
    one = torch.zeros(b, ns, 1, dtype=torch.bool)
    one[:, 2, 0] = True
    w_one = weights_of(make("softmax"), counts, one)
    assert torch.allclose(w_one[..., 2], torch.ones_like(w_one[..., 2]), atol=1e-6), "single source not weight 1"
    print("3. single source -> weight 1 OK")

    # 4. max-shift: results must be invariant to a constant added to all scores
    shifted = counts + 100.0
    w_a = weights_of(make("softmax"), counts, valid)
    w_b = weights_of(make("softmax"), shifted, valid)
    assert torch.allclose(w_a, w_b, atol=1e-6), "max-shift invariance broken"
    print("4. max-shift invariance OK")

    # 5. full module forward: both modes finite, padded rows zeroed, gradients flow
    for mode in ("none", "softmax"):
        attn = make(mode)
        v = torch.randn(b, nq, attn.v_dim, requires_grad=True)
        l = torch.randn(b, ns, attn.l_dim, requires_grad=True)
        mask_v = torch.zeros(b, nq, dtype=torch.bool)
        mask_l = torch.zeros(b, ns, dtype=torch.bool)
        mask_l[1, 4:] = True                      # padding convention: True = excluded
        out_v, out_l = attn(v, l, mask_v, mask_l)
        assert torch.isfinite(out_v).all() and torch.isfinite(out_l).all(), f"{mode}: non-finite output"
        assert (out_l[1, 4:] == 0).all(), f"{mode}: padded text rows not zeroed"
        # Backprop both directions: the text-direction projections are not reachable from
        # out_v alone, so a one-sided backward would leave them with no gradient tensor.
        (out_v.sum() + out_l.sum()).backward()
        params = list(attn.parameters())
        # Structural assertion: every parameter must receive a gradient tensor. Whether an
        # individual tensor's gradient is exactly zero depends on the draw (a threshold can
        # sit where its surrogate derivative vanishes), so only report that count.
        assert all(p.grad is not None for p in params), f"{mode}: some parameter got no gradient at all"
        nonzero = sum(1 for p in params if p.grad.abs().sum() > 0)
        print(f"5. forward/backward ({mode}) OK — {nonzero}/{len(params)} parameters with non-zero gradient")

    print("SOFTMAX ATTENTION CHECKS PASS")


if __name__ == "__main__":
    main()
