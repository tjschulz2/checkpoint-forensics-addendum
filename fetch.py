"""Download the row slices listed in models.json from pinned Hugging Face revisions into rows/.

Only the needed rows are read, with HTTP range requests against the safetensors files (about 23 MB in total).
    python fetch.py            # fetch any missing slices
    python fetch.py --hashes   # (maintainer) recompute sha256 values in models.json
"""
import hashlib, json, os, struct, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "rows")
MANIFEST = os.path.join(HERE, "models.json")


def _get(url, a=None, b=None):
    headers = {"User-Agent": "checkpoint-forensics-addendum"}
    if a is not None:
        headers["Range"] = f"bytes={a}-{b}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise


def fetch(entry):
    base = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/"
    shard = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"][entry["tensor"]]
    n = struct.unpack("<Q", _get(base + shard, 0, 7))[0]
    meta = json.loads(_get(base + shard, 8, 7 + n))[entry["tensor"]]
    shape, (a0, b0), dtype = meta["shape"], meta["data_offsets"], meta["dtype"]
    row_bytes = (b0 - a0) // shape[0]
    r0, r1 = entry["rows"]
    raw = _get(base + shard, 8 + n + a0 + r0 * row_bytes, 8 + n + a0 + r1 * row_bytes - 1)
    if dtype == "BF16":
        arr = np.frombuffer(raw, np.uint16)          # stored as raw bf16 bits
    elif dtype == "I64":
        arr = np.frombuffer(raw, np.int64).astype(np.int16)  # expert ids < 384
    else:
        raise ValueError(dtype)
    arr = arr.reshape([r1 - r0] + shape[1:])
    np.save(path(entry), arr)
    return arr


def path(entry):
    return os.path.join(ROWS, entry["file"] + ".npy")


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(ROWS, exist_ok=True)
    manifest = json.load(open(MANIFEST))
    for e in manifest:
        if os.environ.get("REFETCH") or not os.path.exists(path(e)):
            print(f"fetching {e['file']} ...", flush=True)
            fetch(e)
        if "--hashes" in sys.argv:
            e["sha256"] = sha256(path(e))
    if "--hashes" in sys.argv:
        with open(MANIFEST, "w") as f:
            f.write("[\n" + ",\n".join("  " + json.dumps(e) for e in manifest) + "\n]\n")
        print("wrote sha256 values to models.json")


if __name__ == "__main__":
    main()
