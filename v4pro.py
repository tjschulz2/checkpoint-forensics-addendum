"""DeepSeek-V4-Pro: (A) exact copy of its untouched rows with DeepSeek's seed at 152 SMs,
(B) CPU-only vendor fingerprint of the sin/cos rounding, using V4-Flash (NVIDIA per your MI300X test) as reference.

    python v4pro.py
"""
import numpy as np
from common import load, normal, draws, bf16, bf16_step, corr

SEED, STD = 953049583, 0.006
MODELS = {"DeepSeek-V4-Flash": (148, 4096), "DeepSeek-V4-Pro": (152, 7168)}
BYTE_ROWS = [127, 128] + list(range(180, 191))
TAIL = range(128800, 129280)


def model_rows(model):
    d = MODELS[model][1]
    head, tail = load(f"{model}.embed.120-200"), load(f"{model}.embed.128800-129280")
    get = lambda r: head[r - 120] if r < 200 else tail[r - 128800]
    return d, get


def untouched(model, sms):
    d, get = model_rows(model)
    keep = []
    for r in BYTE_ROWS + list(TAIL):
        if corr(get(r), normal(SEED, 0, sms, r * d + np.arange(d))) > 0.9999:
            keep.append(r)
    return keep


# ---------- A. exact copy ----------
print("A. Untouched rows reproduced by seed 953049583 (rows with r > 0.9999)")
for model, (true_sms, _) in MODELS.items():
    d, get = model_rows(model)
    cand = BYTE_ROWS + list(TAIL)
    line = []
    for sms in (132, 148, 152):
        n = sum(corr(get(r), normal(SEED, 0, sms, r * d + np.arange(d))) > 0.9999 for r in cand)
        line.append(f"{sms} SMs: {n:3d}/{len(cand)}")
    r = np.array(BYTE_ROWS)
    byte_r = corr(np.stack([get(x) for x in r]), normal(SEED, 0, true_sms, r[:, None] * d + np.arange(d)))
    plus1 = corr(np.stack([get(x) for x in r]), normal(SEED + 1, 0, true_sms, r[:, None] * d + np.arange(d)))
    print(f"  {model:18s} " + "   ".join(line) + f"   | byte rows r={byte_r:.6f}, seed+1 r={plus1:+.4f}")

# ---------- B. vendor fingerprint ----------
f32 = np.float32
INV, HALF = f32(2.0 ** -32), f32(2.0 ** -33)
INV2PI = f32(INV * f32(6.2831855)); HALF2PI = f32(INV2PI / f32(2))
REV = f32(0.15915494)


def pipeline(sms, idx, mode):
    """cuRAND _curand_box_muller in float32 with the sin/cos argument handled per `mode`, then *std, then bf16."""
    a, b, c = draws(SEED, 0, sms, idx)
    u = (a.astype(f32) * INV + HALF).astype(f32)
    v = (b.astype(f32) * INV2PI + HALF2PI).astype(f32)
    s = np.sqrt(-2.0 * np.log(u.astype(np.float64))).astype(f32)
    if mode == "ideal":
        ang = v.astype(np.float64)
    else:  # scale to revolutions in float32 first, then sin/cos of 2*pi*r
        p = v.astype(np.float64) * np.float64(REV); r = p.astype(f32)   # RNE: AMDGPU LowerTrig (plain FMUL)
        if mode == "RZ":    # NVIDIA __sinf/__cosf: FMUL.RZ by 1/(2*pi), then MUFU.SIN/COS
            r = np.where(np.abs(r.astype(np.float64)) > np.abs(p), np.nextafter(r, f32(0)), r)
        if mode == "RA":    # control: round away from zero
            r = np.where(np.abs(r.astype(np.float64)) < np.abs(p), np.nextafter(r, f32(np.inf)), r)
        ang = 2 * np.pi * r.astype(np.float64)
    trig = np.where(c % 2 == 0, np.sin(ang), np.cos(ang)).astype(f32)
    return bf16((s * trig).astype(f32) * f32(STD)), v, c, trig


print("\nB. Where each model's hardware sin/cos error pushes a value across a bf16 boundary ('flips', clean rows only)")
print(f"  {'model':18s} {'values':>10s} {'flips/10k':>9s} {'near 0-cross':>12s} {'sin +':>6s} {'cos +':>6s}"
      f"   mismatches: ideal / RNE / RZ / RA   RZ vs RNE: fixes, breaks, sign-test z")
for model, (sms, _) in MODELS.items():
    d, get = model_rows(model)
    keep = untouched(model, sms)
    pub = np.stack([get(r) for r in keep])
    idx = (np.array(keep)[:, None] * d + np.arange(d)).ravel()
    ideal, v, c, trig = pipeline(sms, idx, "ideal")
    ideal = ideal.reshape(pub.shape)
    # drop rows that training touched lightly: any multi-step mismatch, or a mismatch count far above the median
    mis = (ideal != pub).sum(1); multi = (np.abs(pub - ideal) > 1.01 * bf16_step(ideal)).sum(1)
    clean = (multi == 0) & (mis <= 4 * np.median(mis) + 3)
    sel = np.repeat(clean, d)
    P = pub[clean].ravel(); I = ideal[clean].ravel(); V, C, TR = v[sel], c[sel], trig[sel]
    flip = P != I; sgn = np.sign(P - I)
    sinf, cosf = flip & (C % 2 == 0), flip & (C % 2 == 1)
    zc = np.where(C % 2 == 0, np.minimum.reduce([np.abs(V - a) for a in (0, np.pi, 2 * np.pi)]),
                  np.minimum(np.abs(V - np.pi / 2), np.abs(V - 1.5 * np.pi)))
    miss = {m: pipeline(sms, idx[sel], m)[0] != P for m in ("ideal", "RNE", "RZ", "RA")}
    fixes, breaks = int((miss["RNE"] & ~miss["RZ"]).sum()), int((~miss["RNE"] & miss["RZ"]).sum())
    z = (fixes - breaks) / np.sqrt(fixes + breaks)
    print(f"  {model:18s} {P.size:10,d} {flip.sum() / P.size * 1e4:9.2f} {np.mean(zc[flip] < 0.05):12.2f} "
          f"{np.mean(sgn[sinf] > 0):6.2f} {np.mean(sgn[cosf] > 0):6.2f}   "
          f"{' / '.join(str(int(miss[m].sum())) for m in ('ideal', 'RNE', 'RZ', 'RA'))}"
          f"   {fixes}, {breaks}, {z:.1f}   ({clean.sum()}/{len(keep)} rows clean)")
print("\n  'sin +' / 'cos +' = share of flips where the hardware result was above the ideal one.")
print("  RNE / RZ / RA = angle scaled to revolutions in float32 with round-to-nearest (AMD's lowering),")
print("  round-toward-zero (NVIDIA's FMUL.RZ before MUFU.SIN/COS) or round-away (control) before sin/cos.")
