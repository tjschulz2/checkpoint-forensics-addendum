"""Head-to-head NVIDIA vs AMD model of the Box-Muller inputs, exact in float32, on untouched rows.

cuRAND  (curand_normal.h):   u = x*2^-32 + 2^-33          v = y*K + K/2        then __sincosf(v)
rocRAND (rocrand_normal.h):  u = fmaf(x, 2^-32, 2^-32)    v = fmaf(y, K, K)    then __sincosf(v)
K = 2^-32 * 6.2831855f is bit-identical in both (0x30c90fdb).
The prescale before the sine unit is round-toward-zero on NVIDIA (FMUL.RZ) and round-to-nearest on AMD.
The sine units themselves are not modeled (ideal sin/cos), so neither model is bit-exact.

    python vendor_exact.py
"""
import numpy as np
from common import load, normal, draws, bf16, bf16_step, corr

SEED, STD = 953049583, 0.006
MODELS = {"DeepSeek-V4-Flash": (148, 4096), "DeepSeek-V4-Pro": (152, 7168)}
BYTE_ROWS = [127, 128] + list(range(180, 191))
f32 = np.float32
INV, HALF = f32(2.0 ** -32), f32(2.0 ** -33)
K = f32(INV * f32(6.2831855)); MK = np.uint64(0xC90FDB)   # K = MK * 2^-53
REV = f32(0.15915494)


def fma_v(y, full_addend):
    """Exact float32 fma(y_f, K, c) with c = K (AMD) or K/2 (NVIDIA with contraction), via integers in units of 2^-54."""
    yf = y.astype(f32).astype(np.uint64)                       # uint -> float conversion, then exact integer
    T = np.uint64(2) * yf * MK + (np.uint64(2) * MK if full_addend else MK)
    e = np.frexp(T.astype(np.float64))[1].astype(np.int64)     # bit length, fixed up for float64 rounding
    e = np.where((T >> (e - 1).astype(np.uint64)) == 0, e - 1, e)
    e = np.where((e < 64) & ((T >> np.minimum(e, 63).astype(np.uint64)) != 0), e + 1, e)
    sh = np.maximum(e - 24, 0).astype(np.uint64)
    q = T >> sh
    rem = T & ((np.uint64(1) << sh) - np.uint64(1))
    half = np.where(sh > 0, np.uint64(1) << np.maximum(sh, 1).astype(np.uint64) - np.uint64(1), np.uint64(0))
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & np.uint64(1)) == 1)))
    q = q + up.astype(np.uint64)
    return np.ldexp(q.astype(np.float64), sh.astype(np.int64) - 54).astype(f32)


def predict(sms, idx, u_mode, v_mode, prescale):
    a, b, c = draws(SEED, 0, sms, idx)
    xf = a.astype(f32)
    u = (xf * INV + (INV if u_mode == "amd" else HALF)).astype(f32)     # product is exact, so fma == mul+add here
    if v_mode == "nv":       v = ((b.astype(f32) * K).astype(f32) + f32(K / f32(2))).astype(f32)
    elif v_mode == "nv_fma": v = fma_v(b, False)
    else:                    v = fma_v(b, True)
    s = np.sqrt(-2.0 * np.log(u.astype(np.float64))).astype(f32)
    p = v.astype(np.float64) * np.float64(REV); r = p.astype(f32)
    if prescale == "RZ":
        r = np.where(np.abs(r.astype(np.float64)) > np.abs(p), np.nextafter(r, f32(0)), r)
    ang = 2 * np.pi * r.astype(np.float64)
    trig = np.where(c % 2 == 0, np.sin(ang), np.cos(ang)).astype(f32)
    return bf16((s * trig).astype(f32) * f32(STD))


def clean_rows(model, sms, d):
    head, tail = load(f"{model}.embed.120-200"), load(f"{model}.embed.128800-129280")
    get = lambda r: head[r - 120] if r < 200 else tail[r - 128800]
    keep = [r for r in BYTE_ROWS + list(range(128800, 129280))
            if corr(get(r), normal(SEED, 0, sms, r * d + np.arange(d))) > 0.9999]
    pub = np.stack([get(r) for r in keep]); idx = (np.array(keep)[:, None] * d + np.arange(d))
    ref = predict(sms, idx.ravel(), "nv", "nv", "RZ").reshape(pub.shape)
    mis = (ref != pub).sum(1); multi = (np.abs(pub - ref) > 1.01 * bf16_step(ref)).sum(1)
    ok = (multi == 0) & (mis <= 4 * np.median(mis) + 3)
    return pub[ok].ravel(), idx[ok].ravel()


VARIANTS = {  # name: (u, v, prescale)
    "NVIDIA (mul+add, RZ)": ("nv", "nv", "RZ"),
    "NVIDIA (fma, RZ)":     ("nv", "nv_fma", "RZ"),
    "AMD (fma, K addend, RNE)": ("amd", "amd", "RNE"),
    "AMD inputs + RZ prescale": ("amd", "amd", "RZ"),
    "NVIDIA inputs + RNE prescale": ("nv", "nv", "RNE"),
}
for model, (sms, d) in MODELS.items():
    P, idx = clean_rows(model, sms, d)
    pred = {k: predict(sms, idx, *v) for k, v in VARIANTS.items()}
    print(f"\n{model}: {P.size:,} values on clean untouched rows")
    for k in VARIANTS:
        print(f"  {k:30s} mismatches vs published: {int((pred[k] != P).sum())}")
    nv = min(("NVIDIA (mul+add, RZ)", "NVIDIA (fma, RZ)"), key=lambda k: (pred[k] != P).sum())
    for label, a, b in [("full models", nv, "AMD (fma, K addend, RNE)"),
                        ("input step only (same RZ prescale)", nv, "AMD inputs + RZ prescale")]:
        diff = pred[a] != pred[b]
        na, nb = int((diff & (pred[a] == P)).sum()), int((diff & (pred[b] == P)).sum())
        print(f"  head-to-head, {label}: models differ on {int(diff.sum())} values -> "
              f"published matches NVIDIA {na}, AMD {nb}, neither {int(diff.sum()) - na - nb}")
