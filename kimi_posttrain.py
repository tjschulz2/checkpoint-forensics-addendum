"""Kimi re-draws its reserved-token rows (163586-163839) with a separate, fixed-seed normal_ call.

1. In the K2 lineage the re-draw comes after base pretraining: K2-Base holds near-zero noise (std ~3e-5) that is
   uncorrelated with K2-Instruct's full-size values.
2. K3's block does not use the report's K3 seed: 20256586 at offset 3776 and 152 SMs reproduces K3's byte rows but
   gives ~0 on this block.
3. K2.5 and K3 share the block's seed. Agreement runs for exactly 132 x 2048 values from row 163586, then realigns at
   shifts of 1, 2 and 3 x 20 SMs, so K2.5's re-draw ran on 132 SMs and K3's on 152. If K3 follows the K2 pattern,
   that is 152-SM hardware after base pretraining. This is inferred: K3 has no public Base model.
    python kimi_posttrain.py
"""
import numpy as np
from common import load, normal, corr

START, END = 163586, 163840

# 1) zero in base, random after post-training
b = load("Kimi-K2-Base.embed.163624-163664")
i = load("Kimi-K2-Instruct.embed.163624-163664")
print(f"1. Reserved rows 163624-163663: K2-Base std {b.std():.1e} vs K2-Instruct std {i.std():.1e}; "
      f"correlation between them r = {corr(b, i):+.4f} (re-drawn, not rescaled)")

# 2) K3's block is a different init event from its pretraining rows
k3_byte_rows = np.array([124, 125] + list(range(177, 188)))
byte = load("Kimi-K3.embed.124-188")[k3_byte_rows - 124]
blk = load("Kimi-K3.embed.163586-163840")
r_byte = corr(byte, normal(20256586, 3776, 152, k3_byte_rows[:, None] * 7168 + np.arange(7168)))
sample = np.arange(163700, 163720)
r_blk = corr(blk[sample - START], normal(20256586, 3776, 152, sample[:, None] * 7168 + np.arange(7168)))
print(f"2. Seed 20256586/offset 3776/152 SMs: K3 byte rows r = {r_byte:.6f}; K3 reserved block r = {r_blk:+.4f}")

# 3) K2.5 vs K3 on the block (block-relative index j = (row - 163586) * 7168 + col)
A = load("Kimi-K2.5.embed.163586-163840").ravel(); B = blk.ravel()
c = np.median(B[100_000:120_000] / A[100_000:120_000])  # scale between the two (untouched region)
agree = lambda x, y: np.abs(y - c * x) <= 0.02 * np.abs(c * x) + 2e-3
same = agree(A, B)
G = 132 * 2048
row, col = divmod(G, 7168)
print(f"3. Same-index agreement: {same[:G - 2000][150_000:].mean():.3f} just before j = {G:,} "
      f"(= 132 x 2048, row {START + row} col {col}), {same[G + 2000:G + 40_000].mean():.3f} just after")
D = 20 * 2048
for k, name in [(1, ".y"), (2, ".z"), (3, ".w")]:
    lo, hi = k * G + 2000, min((k + 1) * G - 2000, len(A) - k * D)
    ctrl = agree(A[lo:hi], B[lo + (k % 3 + 1) * D:hi + (k % 3 + 1) * D]).mean() if hi + 4 * D < len(A) else float("nan")
    print(f"   {name} region shifted by {k} x 20 SMs: {agree(A[lo:hi], B[lo + k * D:hi + k * D]).mean():.3f} agree "
          f"(control, wrong multiple: {ctrl:.3f})")
print("   -> K2.5's reset: 132 SMs (Hopper). K3's reset: 152 SMs.")
