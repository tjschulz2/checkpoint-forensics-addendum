# checkpoint-forensics addendum

An independent analysis of which GPUs created five open-weight models: DeepSeek-V4-Flash, DeepSeek-V4-Pro, Kimi K2.5, Kimi K3 and Kimi K3.1-Pro. It reads the evidence directly from their published weights.

Most of the hardware findings below were reached before we read Will DePue's report, [Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs](https://unemploymentlabs.com/checkpoint-forensics) ([code](https://github.com/macrogrants/checkpoint-forensics)), published November 12, 2026. Where they overlap, the two analyses agree. This repository documents both what we found independently and what we added after reading the report.

**Quick start.** Each script runs offline in a few seconds:

    pip install numpy
    python v4pro.py           # findings 1 and 2
    python tid2eid_layout.py  # finding 3
    python kimi_posttrain.py  # finding 4

---

## Summary

| # | Finding | Status | Found before reading the report? |
|---|---|---|---|
| 1 | DeepSeek-V4-Pro's untouched rows are an exact copy of its initialization at **152 SMs**. The report rated V4-Pro "partial survival, 49σ". | Exact (r = 0.999999) | No: it uses the report's published DeepSeek seed |
| 2 | V4-Pro's floating-point rounding matches **NVIDIA's** sine/cosine code path, not AMD's (paired z = 7.6) | Statistical, not bit-exact | No |
| 3 | DeepSeek's untrained hash-routing tables put V4-Flash on **148 SMs** and V4-Pro on **152 SMs**, with no seed needed | Exact layout match | **Yes** |
| 4 | Kimi re-draws its reserved-token rows with a fixed seed. That re-draw ran on **132 SMs** for K2.5 and **152 SMs** for K3. | Exact layout match; timing inferred | **Yes** |

By standard SM counts, 132 is the Hopper class (H100, H800, H200), 148 is B200/B300, and 152 is GB200/GB300. An AMD MI300X can also be configured to 152 or 148; finding 2 addresses that.

---

## Background: why initial weights record the GPU

Before training, a model's weights are filled with random numbers from PyTorch's `normal_`. On NVIDIA GPUs this uses the Philox4x32-10 generator through cuRAND. PyTorch splits the job across a fixed number of GPU threads, set in [`DistributionTemplates.h`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistributionTemplates.h):

    threads G = multiProcessorCount × (maxThreadsPerMultiProcessor / 256) × 256   = 2048 × SMs on these GPUs

The flat element `i` is written by thread `t = i mod G`. Each thread produces four values per call (`.x .y .z .w`): component `c = (i // G) mod 4`, and call `k = i // 4G`. Every thread gets its own Philox stream from the seed. So the same seed run on GPUs with different SM counts produces the same numbers, but in different places.

Some weights are never trained, so they keep these values exactly:

- embedding rows for tokens that never appear in the data
- DeepSeek's fixed hash-routing tables

Those values record the thread count `G`, and therefore the SM count of the machine that generated them.

There are two ways to read `G`:

- **With the seed:** regenerate the values for each candidate SM count and see which one matches. This is the report's approach.
- **Without the seed:** if two models were generated from the **same seed on different machines**, their values agree position by position up to the smaller `G`, and then stop. Past that point, component `c` of thread `t` lives at `c·G + t`, so the two realign at shifts of exactly `c × (G_large − G_small)`. The cutoff gives one thread count and the shift gives the other. This is the approach we used first.

---

## What we did, in order

### Before reading the report

All of this was done on November 13, 2026, without access to the report or its seeds.

1. **Architecture and lineage from configs and tensor maps.** We read configs and safetensors headers by HTTP range request, without downloading full weights. Kimi K2.5 uses DeepSeek-V3's architecture. K3 is a new design. K3.1-Pro is K3 plus exactly 120 new tensors: a DeepSeek-style sparse-attention indexer on each of K3's 24 full-attention layers. Weight comparisons showed K3.1-Pro is K3 trained further (cosine 0.99+), and that K3 was trained from scratch, not from K2.5 (cosine ≈ 0).
2. **First hardware guess (wrong).** From priors and the model cards, we first guessed Nvidia Hopper for all five models. The weights later overturned this.
3. **Hash-routing tables (finding 3).** DeepSeek V4's `tid2eid` tables have the form `topk(rand(vocab, experts), 6)`, and they are never trained. Flash and Pro share a seed. Their agreement stops at exactly 148 × 2048 and realigns at a 4-SM shift, which gives V4-Flash 148 SMs and V4-Pro 152. All seven other V4 releases have byte-identical tables. V4.1-Flash has no tables and no detectable signal.
4. **Seed search (failed).** We implemented Philox (checked against the Random123 known-answer test) and the cuRAND/PyTorch layout. We then searched seeds up to 4.2 million, plus a range of offsets, for the tables. We found nothing; DeepSeek's real seed is about 953 million.
5. **Kimi reserved rows (finding 4).** In Kimi models, rows 163586–163839 are near-zero in base models and full-size random values after post-training. K2.5 and K3 share the seed for these rows. The layout gives 132 SMs for K2.5's re-draw and 152 for K3's. K2-Instruct and Kimi-Linear-Instruct share a different seed with equal thread counts, so the same GPU type, which we could not identify.
6. **Spec lookups.** We matched SM counts to products using public spec sheets only; we did no searches about the labs.
   - 132 is Hopper; 148 is B200.
   - We could not find a product with 152.
   - We wrongly ruled out AMD MI300X, missing that it can be partitioned to 152.

### After reading the report

7. **Independent replication.** Our own Philox code reproduces the report's DeepSeek seed (953049583) and K3 seed (20256586, offset 3776). Every SM count we measured in steps 3 and 5 matches the report. The report also resolved our two gaps: 152 = GB200/GB300, and the MI300X caveat.
8. **V4-Pro exact copy (finding 1).** We applied DeepSeek's seed to V4-Pro's untouched rows.
9. **Vendor test (finding 2).** The report's physical AMD test covered only V4-Flash. We built a CPU-only alternative from the vendors' public code paths, with controls.
10. **Re-checks.** We checked each claim against the full report text, which led to these adjustments:
    - The K3 timing in finding 4 is inferred, not proven.
    - Finding 3 is a confirmation, not new evidence, once finding 1 exists.

---

## Findings

### 1. DeepSeek-V4-Pro is an exact copy at 152 SMs (`v4pro.py`, part A)

**Method.** Regenerate `normal_(std = 0.006)` with seed 953049583 for each candidate SM count. Compare against V4-Pro's byte-token rows (127, 128, 180–190) and padded tail rows (128800–129279), which are the same rows the report uses for V4-Flash.

| Model | 132 SMs | 148 SMs | 152 SMs | Byte rows r | Seed+1 |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 0/493 | 445/493 | 0/493 | 0.999999 | +0.005 |
| DeepSeek-V4-Pro | 0/493 | 0/493 | **449/493** | **0.999999** | −0.004 |

(Rows count as matching at r > 0.9999. The other candidate rows were changed by training, some lightly and some fully.)

**What this adds.** The report lists V4-Pro (and V4-Pro-Base, whose rows are byte-identical) as partial survival at 49σ, its weakest Blackwell result. V4-Pro is in fact an exact copy, as strong as V4-Flash. It is initialized the same way as V4-Flash (float32, std 0.006, same seed and rows), so the report's physical MI300X value test, which settled AMD for V4-Flash, can be applied to V4-Pro unchanged.

### 2. V4-Pro's rounding matches NVIDIA, not AMD (`v4pro.py`, part B)

**Why vendors differ.** cuRAND's and rocRAND's Box-Muller both call `__sincosf`, but they compile differently:

- **NVIDIA:** `FMUL.RZ` by 1/(2π), which rounds toward zero, then `MUFU.SIN`/`MUFU.COS` ([disassembly](https://zenn.dev/toropippi/articles/016f3f36f8c6ce), [NVIDIA forums](https://forums.developer.nvidia.com/t/sm-70-and-sm-75-with-sinf-cosf-and-sincosf/110013)).
- **AMD:** HIP's `__sincosf` → `__ocml_native_sin/cos_f32`. LLVM's AMDGPU `LowerTrig` scales with a plain `FMUL`, which rounds to nearest, then `v_sin`/`v_cos` ([source](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIISelLowering.cpp)).

**Method.**

1. Rebuild cuRAND's float32 Box-Muller (`u = x·2⁻³² + 2⁻³³`, `v = y·2π·2⁻³² + π·2⁻³²`, `s = sqrt(−2 ln u)`), then `× 0.006`, then round to bf16.
2. Model the sin/cos angle four ways: ideal, round-to-nearest prescale (AMD), round-toward-zero prescale (NVIDIA), and round-away prescale (a control).
3. Count where the published bf16 values differ from each model. Each difference ("flip") is a place where the hardware's sin/cos error pushed a value across a bf16 boundary.

Only clean rows are used: rows with any multi-step mismatch, a sign of light training, are dropped. V4-Flash, which the report showed is NVIDIA, is the reference.

| | V4-Flash (NVIDIA) | V4-Pro |
|---|---|---|
| Values compared | 1,699,840 | 2,874,368 |
| Flips per 10k (vs ideal) | 2.28 | 2.31 |
| Flips within 0.05 rad of a sin/cos zero-crossing | 0.67 | 0.67 |
| Cos flips where hardware > ideal | 0.24 | 0.26 |
| Mismatches: ideal / RNE (AMD) / **RZ (NVIDIA)** / RA (control) | 387 / 328 / **225** / 404 | 663 / 572 / **424** / 654 |
| RZ vs RNE, paired: values fixed / broken (sign-test z) | 175 / 72 (6.6) | 265 / 117 (**7.6**) |

**Result.** V4-Pro matches the NVIDIA reference on every measure. Its values fit NVIDIA's round-toward-zero prescale much better than AMD's round-to-nearest (7.6σ, paired). The round-away control fits worse than ideal math, so the effect is specific to round-toward-zero, not a generic gain from perturbing the angle.

**Limits.**

- This is not bit-exact. The model explains about 40% of flips; the rest come from each sine unit's own approximation, which we cannot emulate.
- We observed no AMD output directly.
- [SFU-Sim](https://github.com/hzX40000000/SFU-Sim) claims a bit-exact NVIDIA MUFU model, but it ships as a precompiled Linux library, does not cover datacenter Blackwell (sm_100), and was not used here.

### 3. Hash-routing tables give 148 and 152 without a seed (`tid2eid_layout.py`)

**Method.** `layers.{0,1,2}.ffn.gate.tid2eid` holds the six experts per token for DeepSeek's three hash-routed layers. It is never trained, and Flash (256 experts) and Pro (384) generated it from the same seed. A Flash row (256 scores) sits inside a Pro row (384 scores) at the same flat position. The tables are low-precision, so ties reorder; we therefore compare **sets**: Pro's picks that fall in Flash's range must be among Flash's picks.

**Result.**

- **Cutoff.** Same-position agreement is 0.999 below flat index 302,848 and 0 from 303,104. The only `2048 × SMs` in that window is 148 × 2048.
- **Shift.** Scanning shifts in the `.y` region, only +4 SMs matches (150/150; the next best is 1/150). That gives V4-Pro 152 × 2048.
- **Every region, every layer:**

| Layer | .x | .y | .z | .w | 2nd call | Control |
|---|---|---|---|---|---|---|
| 0 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 0/300 |
| 1 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 1/300 |
| 2 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | 0/300 |

**What this adds.** It independently confirms findings 1 and 2, using a different tensor and no seed. We reached it before seeing the report.

### 4. Kimi's fixed-seed reserved-row re-draw: 132 SMs in K2.5, 152 in K3 (`kimi_posttrain.py`)

**Method and result.**

- **The re-draw is a separate call.** Rows 163586–163839 are written by their own `normal_` call: the layout starts exactly at row 163586.
- **In the K2 lineage it happens after base pretraining.** K2-Base holds near-zero noise (std 3e-5), uncorrelated with K2-Instruct's full-size values (r = −0.0006).
- **K3's block is not from K3's main initialization.** The report's K3 seed reproduces K3's byte rows (r = 0.999996) but gives r = +0.003 on this block.
- **K2.5 and K3 share the block's seed.** Same-position agreement is 0.996 for exactly 132 × 2048 values, ending at row 163623, column 5120. It then drops to 0.07, and the `.y/.z/.w` regions realign at 1, 2 and 3 × 20 SMs (0.99 / 1.00 / 1.00 vs 0.07 controls). So K2.5's re-draw ran on 132 SMs and K3's on 152.

**What this adds.** If K3 follows the K2 pattern, this places 152-SM hardware **after** K3's base pretraining, not only at initialization. That speaks to the report's caveat that initialization need not imply training.

**Limits.** The timing is inferred: K3 has no public base model. The report notes at least 64 separately seeded 152-SM processes in K3, and this re-draw may be one of them.

---

## What this does not show

- Where the chips were, who owned them, or how they were obtained. Blackwell systems can be rented outside China.
- That an entire training run used the identified hardware. These are initialization fingerprints, plus the inferred later stage in finding 4.
- Which chip within a class: H100/H800/H200 share 132 SMs; B200/B300 share 148; GB200/GB300 share 152.

## Repository layout

| File | Purpose |
|---|---|
| `models.json` | Every row slice: Hugging Face repo, pinned revision, tensor, rows and sha256 |
| `fetch.py` | Downloads the slices into `rows/` with HTTP range requests (`REFETCH=1` to redo) |
| `rows/` | The 15 slices used (21 MB); hashes are checked on every load |
| `common.py` | Hash-checked loading, Philox4x32-10 (KAT-checked on import) and the `normal_` layout |
| `v4pro.py` | Findings 1 and 2 |
| `tid2eid_layout.py` | Finding 3 |
| `kimi_posttrain.py` | Finding 4 |

## Credits

Will DePue's report and code established the method's calibration, published the DeepSeek and Kimi seeds used in findings 1 and 2 and in finding 4's control, and identified 152 SMs as GB200/GB300. Findings 3 and 4 were reached independently, before we read the report.

MIT License.
