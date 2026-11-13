"""Shared helpers: hash-checked row loading, Philox4x32-10, and the PyTorch CUDA normal_ layout."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_manifest = {e["file"]: e for e in json.load(open(os.path.join(HERE, "models.json")))}


def load(name):
    """Rows saved by fetch.py, checked against the sha256 in models.json. bf16 slices come back as float32.
    Missing slices (or REFETCH=1) are downloaded from the pinned Hugging Face revision first."""
    import fetch
    e = _manifest[name]; p = fetch.path(e)
    if os.environ.get("REFETCH") or not os.path.exists(p):
        os.makedirs(fetch.ROWS, exist_ok=True); fetch.fetch(e)
    if "sha256" in e and fetch.sha256(p) != e["sha256"]:
        raise SystemExit(f"{name}: sha256 mismatch with models.json")
    a = np.load(p)
    return (a.astype(np.uint32) << 16).view(np.float32) if a.dtype == np.uint16 else a


# ---- Philox4x32-10 (cuRAND / Random123 layout) ----
_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
_MASK = np.uint64(0xFFFFFFFF)


def philox(c0, c1, c2, c3, k0, k1):
    c0, c1, c2, c3 = (np.asarray(x, np.uint64) & _MASK for x in (c0, c1, c2, c3))
    k0, k1 = np.asarray(k0, np.uint64) & _MASK, np.asarray(k1, np.uint64) & _MASK
    for r in range(10):
        p0, p1 = _M0 * c0, _M1 * c2
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ k0) & _MASK, p1 & _MASK, ((p0 >> np.uint64(32)) ^ c3 ^ k1) & _MASK, p0 & _MASK
        if r < 9:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c0, c1, c2, c3


assert [int(x) for x in philox(0, 0, 0, 0, 0, 0)] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8], "Philox KAT failed"


def draws(seed, offset, sms, idx):
    """Raw Philox words feeding element idx of one torch.cuda normal_ call (grid-stride, G = 2048*sms threads).
    Element i: thread t = i % G, component c = (i // G) % 4, call k = i // (4G)."""
    G = 2048 * sms
    idx = np.asarray(idx, np.int64)
    t = (idx % G).astype(np.uint64); q = idx // G; c = q % 4
    ctr = np.uint64(offset // 4) + (q // 4).astype(np.uint64)
    x, y, z, w = philox(ctr & _MASK, ctr >> np.uint64(32), t & _MASK, t >> np.uint64(32),
                        np.uint64(seed) & _MASK, np.uint64(seed) >> np.uint64(32))
    return np.where(c < 2, x, z), np.where(c < 2, y, w), c


def normal(seed, offset, sms, idx):
    """Standard-normal value (float64 Box-Muller, cuRAND constants) written at idx."""
    a, b, c = draws(seed, offset, sms, idx)
    inv = 2.0 ** -32
    s = np.sqrt(-2 * np.log(a * inv + inv / 2))
    v = b * inv * 2 * np.pi + inv * np.pi
    return np.where(c % 2 == 0, s * np.sin(v), s * np.cos(v))


def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)) << np.uint64(16)
    return r.astype(np.uint32).view(np.float32)


def bf16_step(x):
    x = np.abs(np.asarray(x, np.float32))
    return 2.0 ** (np.floor(np.log2(np.maximum(x, 1e-30))) - 7)


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
