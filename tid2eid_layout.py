"""DeepSeek V4 hash-routing tables (layers.{0,1,2}.ffn.gate.tid2eid) are never trained, and Flash and Pro were
generated from the same seed. That gives a seed-free layout reading for V4-Pro:
Flash flat index i and Pro flat index i hold the same Philox thread's .x value only while i < G_small; past that,
component c of thread t sits at c*G + t, so the two tables realign at a shift of c * (G_pro - G_flash).

The tables look like topk(rand(vocab, n_experts), 6) in low precision, so only sets are compared (ties reorder).
    python tid2eid_layout.py
"""
import numpy as np
from common import load

F = {l: load(f"DeepSeek-V4-Flash.tid2eid.{l}") for l in range(3)}  # first 6,000 of 129,280 rows; 256 experts
P = {l: load(f"DeepSeek-V4-Pro.tid2eid.{l}") for l in range(3)}    # first 6,000 rows; 384 experts


def consistent(f_row, p_row, off):
    """Pro row covers flat [384r, 384r+384); Flash row sits at column offset `off` inside it."""
    s = [x - off for x in p_row if off <= x < off + 256]
    return bool(s) and set(s) <= set(f_row)


def test(l, region, shift, gs, n=300):
    """Flash rows in region [region*gs, (region+1)*gs) vs Pro at flat + shift."""
    f, p = F[l], P[l]; hits = tot = 0
    for r in range((region * gs) // 256 + 2, ((region + 1) * gs) // 256 - 2):
        pr, off = divmod(r * 256 + shift, 384)
        if off + 256 > 384 or not any(off <= x < off + 256 for x in p[pr]):
            continue
        tot += 1; hits += consistent(f[r], p[pr], off)
        if tot == n:
            break
    return hits, tot


# 1) where does same-index agreement stop?
f, p = F[0], P[0]
ok = [(768 * k, consistent(f[3 * k], p[2 * k], 0)) for k in range(700)] + \
     [(768 * k + 512, consistent(f[3 * k + 2], p[2 * k + 1], 128)) for k in range(700)]
ok.sort()
last = max(i for i, g in ok if g and i < 400_000)
nxt = min(i for i, g in ok if i > last)
rate_before = np.mean([g for i, g in ok if i < last])
print(f"Same-index agreement: {rate_before:.3f} of rows below flat {last + 256:,}, 0 beyond; first failing row starts at {nxt:,}.")
cands = [s for s in range(16, 257) if last + 256 <= 2048 * s <= nxt]
print(f"  -> smaller run's thread count G in ({last + 256:,}, {nxt:,}]: only 2048 x {cands} fits (V4-Flash).")

# 2) shifted realignment: scan extra-SM counts on the .y region
gs = 148 * 2048
scores = sorted(((test(0, 1, 2048 * m, gs, n=150)[0], m) for m in range(1, 60)), reverse=True)[:3]
print(f"\n.y-region realignment, best extra-SM shifts (hits of 150): {scores}")
m = scores[0][1]; gl = gs + 2048 * m
print(f"  -> V4-Pro G = 2048 x {gl // 2048}")

# 3) every region, all three layers, with controls
print("\nPredicted shift c*(G_pro - G_flash) per region; control = wrong multiple")
for l in range(3):
    parts = [f"L{l}"]
    for c in range(4):
        parts.append(f".{'xyzw'[c]} {test(l, c, c * (gl - gs), gs)[0]}/300")
    parts.append(f"2nd call .x {test(l, 4, 4 * (gl - gs), gs)[0]}/300")
    parts.append(f"control .z@1x {test(l, 2, gl - gs, gs)[0]}/300")
    print("  " + "  ".join(parts))
