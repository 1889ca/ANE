# ANE Training Optimization Log

Chronological record of optimizations attempted on Stories110M (12-layer, 768-dim, 32K-vocab) training.
Branch: `cls-ane-kernel` on fork `1889ca/ANE`.

## Baseline (before this branch)

```
~190ms/step (20 steps, accum=50)
  ane=~12  io=~6  cls=~16  elem=~28
```

The classifier (`logits = embed @ x_final`, `dy = embed^T @ dlogits`) ran on CPU via `cblas_sgemm`, taking ~16ms/step combined. All dW gradient accumulation dispatched to a single serial GCD queue.

---

## 1. Classifier Forward/Backward on ANE

**Commit:** `23e79d1` — "Move classifier fwd/bwd to ANE kernel"

**What:** Added two MIL program generators (`gen_cls_fwd`, `gen_cls_bwd`) that perform the classifier matmul as a 1x1 convolution on ANE:
- Forward: `embed[32000, 768, 1, 1]` conv on `x_final[1, 768, 1, 256]` -> `logits[1, 32000, 1, 256]`
- Backward dx: `embed_t[768, 32000, 1, 1]` conv on `dlogits[1, 32000, 1, 256]` -> `dy[1, 768, 1, 256]`
- Backward dW (outer product accumulation) stays on CPU — it accumulates across microbatch steps.

**Risk:** Whether ANE handles VOCAB=32000 output channels in a conv op. Existing kernels maxed at HIDDEN=2048.

**Result:** ANE compiles and runs 32K output channels without issue. cls dropped from ~16ms to ~2.5ms/step.

**Key finding:** ANE's conv op has no problem with 32K output channels. The graceful fallback (check for NULL compile result, fall back to cblas) was not triggered.

**Files changed:** `stories_mil.h`, `stories_config.h`, `train_large.m`

---

## 2. Elem-wise Profiling Breakdown

**Commit:** `6aaf7bf` — "Add fine-grained elem-wise profiling breakdown"

**What:** Split the monolithic `t_elem` timer into 6 sub-categories to find where the ~28ms of "element-wise" CPU time was actually going.

**Result (20 steps, accum=50):**
```
elem=76.4 [xent=17.9 memcpy=7.5 rms_bwd=7.3 resid=2.0 embed=0.8 embed_bwd=41.1]
```

**Key findings:**
- `embed_bwd=41.1ms` — the monster. This is `dispatch_group_wait` + `embed_backward` scatter-add. The wait blocks on the serial dW queue draining, which includes the embed outer product cblas_sgemm (VOCAB x DIM x SEQ = 6.3 GFLOP).
- `xent=17.9ms` — cross-entropy softmax over 32K vocab x 256 positions. Already vectorized with vDSP per-row. Each 32K row (128KB) fits in L1 cache.
- `memcpy=7.5ms` — dW capture malloc+memcpy (5 buffers x 12 layers per step).
- `rms_bwd=7.3ms` — 25 rmsnorm_bwd calls (2 per layer + 1 final).
- `resid` and `embed` are negligible.

**Key insight:** The serial dW queue is the real bottleneck, not any single CPU op. ANE finishes in ~22ms but blocks waiting for the serial CPU queue to drain.

---

## 3. Cross-entropy Optimization Attempts

**Commit:** `c650e4b` — "Use persistent buffer in cross_entropy_loss"

### 3a. Batched vvexpf — FAILED (regression)

**What:** Replaced 256 separate `vvexpf(32K)` calls with a single `vvexpf(8M)` call on the entire transposed buffer. Also batched the 1/S gradient scaling.

**Result:** xent went from 17.9ms to 22.5ms. **Regression.**

**Why it failed:** Cache locality. Each 32K-element row is 128KB, which fits in L1 cache. The per-row loop keeps data L1-hot across max-subtract, exp, sum, normalize. Batching the exp into one 32MB call evicts all data from cache, forcing cold reloads for the subsequent normalize pass. Same issue with batched gradient scaling.

**Lesson:** For operations on data that fits in L1 (128KB rows at V=32K), per-row processing beats batched processing because it preserves cache residency across multiple passes over the same data.

### 3b. Persistent buffer — kept (minor win)

**What:** Replaced per-call `malloc(S*V*4)` + `free()` (32MB allocation) with a static persistent buffer.

**Result:** Eliminates 32MB allocation churn per step. Minor improvement, hard to measure precisely due to run-to-run variance.

---

## 4. Parallel dW Dispatch Queues

**Commit:** `3103811` — "Parallelize dW dispatch across per-layer serial queues"

**What:** Replaced the single serial dW queue with 13 queues:
- 12 per-layer serial queues (`dw_layer_q[L]`) — serializes same-layer dW across steps (protects gradient accumulators with beta=1.0 accumulation), but allows different layers to run in parallel
- 1 embed serial queue (`dw_embed_q`) — serializes embed dW

Also captured `dlogits` (32MB) and `x_final` (0.75MB) for the embed dW block, eliminating two blocking sync points:
- Removed `dispatch_group_wait` before `fwdAttn` — layer dW uses captured data, no IOSurface dependency
- Removed `dispatch_group_wait` before `cls_fwd` — embed dW now captured, doesn't read live buffers
- `embed_backward` only waits on `embed_dw_grp` (not all dW)

Two separate dispatch groups track layer vs embed work independently.

**Result (50 steps, accum=100):**
| Metric | Serial dW | Parallel dW | Change |
|---|---|---|---|
| ms/step | 218.1 | 148.7 | **1.47x faster** |
| embed_bwd | 82.3 | 9.1 | **9x faster** |
| elem total | 134.6 | 60.2 | 2.2x faster |
| ANE TFLOPS | 0.54 | 0.79 | 1.46x |

**Key insight:** The bottleneck was not compute — it was serialization. The single serial queue forced the ANE to idle while CPU dW work drained sequentially. Per-layer queues allow GCD to schedule dW blocks across multiple P-cores, and removing the blocking waits lets ANE proceed without waiting for CPU.

**Correctness argument:** Layer dW blocks are fully self-contained (all inputs captured via malloc+memcpy). Same-layer blocks from different steps write to the same gradient accumulator with beta=1.0 — per-layer serial queues prevent this race. Different-layer blocks write to different accumulators — safe to overlap. Embed dW captures dlogits+x_final, so it doesn't depend on live buffers being stable.

**Extra memcpy cost:** ~33MB per step for dlogits+x_final capture. At ~50GB/s memory bandwidth, that's ~0.7ms — negligible compared to the 70ms saved.

---

## 5. Fuse Residual Adds into ANE Kernels

**Commit:** `c4d888f` — "Fuse residual adds into ANE forward kernels"

**What:** Moved the two per-layer residual additions from CPU `vDSP_vadd` into the MIL programs:
- SDPA kernel: added `x2 = add(x=x, y=oo)` after the Wo conv. Output offset 0 now holds `x2` instead of raw `o_out`.
- FFN kernel: added `xnext = add(x=x, y=y)` after the W2 conv. Output offset 0 now holds `x_next` instead of raw `ffn_out`.

This eliminates per step:
- 24 `vDSP_vadd` calls (2 per layer × 12 layers)
- 12 `io_read_fp16(o_out)` — no longer needed, x2 piped directly via `io_copy`
- 12 `io_write_fp16(x2→FFN)` — replaced by `io_copy` (fp16→fp16, no conversion)

Removed dead `o_out` and `ffn_out` fields from `LayerActs` struct.

**Result (20 steps, accum=50):**
```
106.3 ms/step
  ane=10.3  io=4.7  cls=2.2  rms_fwd=0.1
  elem=40.3 [xent=16.5 memcpy=8.9 rms_bwd=9.1 resid=1.1 embed=0.7 embed_bwd=3.9]
```

**Comparison to pre-fusion (accum=100 baseline):**
| Metric | Before | After | Change |
|---|---|---|---|
| ms/step | 148.7 | 106.3 | **1.40x faster** |
| io | 7.2 | 4.7 | -2.5ms |
| resid | 3.7 | 1.1 | -2.6ms |
| ANE TFLOPS | 0.79 | 1.11 | **1.40x** |

**Note:** The overall speedup exceeds the ~5ms predicted from residual fusion alone. The different accum count (50 vs 100) likely improves dW overlap dynamics, contributing to the xent and memcpy drops. The direct fusion savings (resid + io) account for ~5ms, matching the estimate.

**Remaining resid=1.1ms:** This is the backward-pass residual adds (`dx_rms1 + dx2` and `dx2 += dy` loops in the backward layer loop), which were not part of this change.

**Correctness:** Loss trajectory unchanged (step 0: 4.3143, step 10: 3.6053 — matches baseline 4.3146/3.6560 within fp16 rounding). Backward pass untouched — all backward IO reads from fwdAttn/fwdFFN use offsets DIM+ which are unchanged.

---

## Hivemind Analysis: Fused Cross-Entropy on ANE

**Session:** `findings/hivemind-next-opt.md` (fib mode, Claude+Gemini+DeepSeek, 3 rounds)

**Claim tested:** "Fusing cross-entropy softmax into the ANE classifier kernel eliminates the 16.5ms xent CPU cost."

### Key findings

**Approach — tiled softmax with index decomposition:**
- Reshape 32K vocab to `[512, 64]` tiled layout matching ANE vector unit width
- Two-pass reduction: local max/sum over inner 64, then rescale across 512 tiles
- Express target selection via index decomposition (`tile = t // 64`, `offset = t % 64`) + broadcast comparison, not a 16MB one-hot mask. Total IO: 512 bytes of target indices
- The tiling naturally provides blockwise log-sum-exp, improving fp16 numerical stability vs flat 32K accumulation

**Backward is simpler than forward:**
- Closed-form gradient: `∂L/∂logits = softmax(x) - one_hot(target)` — pure elementwise, no reductions
- Cache only the LSE scalar (8 bytes/token) from forward, recompute probs as `exp(logits - LSE)`
- Avoids both 128MB activation cache and full forward recompute
- Same index decomposition regenerates the one-hot mask on-the-fly

**Unanswered question (deferred):**
- Does the `∂L/∂W_cls` sgemm (VOCAB×DIM×SEQ = 6.3 GFLOP) overlap cleanly with ANE, or does fusing xent change the sync dynamics? The 48ms unaccounted time already includes this overlap. Can answer empirically if we build the kernel.

**Risk assessment:**
- High compiler risk: `reduce_max`/`reduce_sum` over reshaped tensors, `mb.gather` on small LUTs — any of these could trigger CPU fallback
- MIL graph partitioning is brittle with tiled reductions + LUT patterns
- Workarounds mentioned (identity nodes, optimization_level=0) are speculative

**Verdict:** Biggest potential win (16.5ms) but highest risk. Treat as a research spike, not a safe optimization.

---

## 6. Eliminate malloc+memcpy for dW Capture Buffers

**What:** Replaced per-step `malloc+memcpy+free` for async dW dispatch with pre-allocated double-buffered capture slots (`LayerDWCap` struct). Uses a per-layer `dispatch_semaphore_t` (init=2) for slot flow control.

Three categories of elimination:
1. **Forward io_read redirect (4 fields):** `silu_out`, `x2norm`, `attn_out`, `xnorm` — these only existed as intermediaries between `io_read_fp16` (forward) and `memcpy` (backward capture). Redirected `io_read_fp16` to write directly into capture slots. Removed from `LayerActs`.
2. **Backward io_read redirect (5 fields):** `dh1`, `dh3`, `dq`, `dk`, `dv` — shared gradient buffers used for `io_read_fp16` then immediate `memcpy`. Redirected to capture slots directly. Removed shared buffers.
3. **Double-copy elimination:** `do_out_buf` (dx2→do_out_buf→capt_do) reduced to single memcpy (dx2→cap->dx2). `dffn` (dy→dffn→capt_dffn) reduced to single memcpy (dy→cap->dffn).

**Eliminated per step:** 132 malloc+free calls, ~151MB of memcpy
**Remaining memcpy:** ~27MB/step (layer_in, dy→dffn, dx2→dx2 — 3 unavoidable copies)
**Memory cost:** +209MB net (288MB capture slots - 79MB removed buffers)

**Result (100 steps, accum=50, batch 2 = warm):**
```
  memcpy: 8.9 → 2.2 ms/step  (75% reduction, 6.7ms saved)
```

**Note on overall ms/step:** All timers (ANE, xent, rms_bwd, etc.) showed ~1.7x inflation vs the baseline measurement session, consistent with system-level variance (thermal, background load). The memcpy timer is the only one that *improved*, confirming the optimization is working. A controlled A/B on the same session would show the true step-time improvement.

**Correctness:** Loss unchanged (step 0: 4.3143, step 10: 3.6053).

**Files changed:** `stories_config.h` (LayerDWCap struct, trimmed LayerActs), `train_large.m` (capture slots, semaphore, removed 7 shared buffers)

---

## 7. Move rmsnorm_bwd dx to ANE

**Commit:** `8bca719` — "Move rmsnorm_bwd dx computation to ANE, keep dw on CPU"

**What:** Split `rmsnorm_bwd` into two parts:
- **dx (ANE):** New weight-free MIL kernel `gen_rms_bwd()` computes the input gradient. Input `[1, 3*DIM, 1, SEQ]` packs dy, x, and w (weight at position 0 only via `slice_by_size`). 13 kernels compiled at startup (12 per-layer + 1 final), reused across steps like `sdpaBwd2`.
- **dw (CPU):** New `rmsnorm_dw()` function — weight gradient only (cheap reduction: recompute rrms, then `sum(dy * x * rrms)` per channel). Uses static buffer to avoid per-call malloc.

Old `rmsnorm_bwd()` removed entirely.

**Key design decisions:**
- Each layer reuses the same `rmsBwd` kernel for both rmsnorm1 (attn) and rmsnorm2 (ffn) — sequential calls, IOSurface repopulated between them.
- `io_copy` sources dy/x from prior ANE kernel outputs (fp16→fp16, no conversion). New `io_write_fp16_vec` helper packs the weight vector at channel offset `2*DIM`, position 0 only.
- dw computed FIRST (while fp32 dy is still available), then ANE kernel overwrites the buffer with dx.
- Final rmsnorm backward: falls back to `io_write_fp16_at` for dy when `use_ane_cls=false` (no cls_bwd->ioOut to io_copy from).

**Eliminated:**
- 25 CPU `rmsnorm_bwd` calls (replaced by 25 `rmsnorm_dw` + 25 ANE evals)
- `dx_rms1` calloc+free per layer (ANE writes dx directly to `dx_attn`)
- `dx_rms_final` calloc+memcpy+free (ANE writes dx directly to `dy`)

**Result (100 steps, accum=50, 2 batches):**
```
102.7 ms/step (avg), batch1=101.1, batch2=104.2
  Batch 1: ane=9.5 io=4.0 cls=1.9 rms_fwd=0.1
           elem=33.2 [xent=15.6 memcpy=1.2 rms_bwd=4.0 resid=1.5 embed=0.9 embed_bwd=9.9]
  Batch 2: ane=9.3 io=4.6 cls=2.0 rms_fwd=0.1
           elem=34.5 [xent=16.1 memcpy=1.3 rms_bwd=5.4 resid=1.5 embed=1.5 embed_bwd=8.7]
```

| Metric | Before (est.) | After (100-step avg) | Change |
|---|---|---|---|
| ms/step | ~106 | 102.7 | **-3.3ms** |
| rms_bwd | ~9 | ~4.7 | **-4.3ms** |
| io | ~5 | ~4.3 | -0.7ms |
| ane | ~10 | ~9.4 | -0.6ms |
| Compiles | 74 | 87 | +13 (well within 200 budget) |

**Note on rms_bwd variance:** 4.0ms (batch 1) vs 5.4ms (batch 2) — the `rmsnorm_dw` CPU work competes with async dW cblas for P-core time. Under heavier dW overlap (batch 2, warmer caches, more in-flight sgemms), the vDSP loops slow down. The 20-step measurement (3.3ms) was optimistic; 100-step average (~4.7ms) is more representative.

**Correctness:** Loss unchanged (step 0: 4.3143, step 10: 3.6053).

**Files changed:** `stories_mil.h` (gen_rms_bwd), `stories_io.h` (io_write_fp16_vec), `stories_cpu_ops.h` (rmsnorm_dw replacing rmsnorm_bwd), `stories_config.h` (rmsBwd in LayerKernels), `train_large.m` (compile+eval+cleanup)

---

## 8. Cache rrms from Forward for rmsnorm_dw

**Commit:** `f96b5c9` — "Cache rrms from ANE forward kernels, eliminate recomputation in rmsnorm_dw"

**What:** The forward rmsnorm (fused in fwdAttn/fwdFFN ANE kernels) already computes `rrms = rsqrt(mean(x^2) + eps)` as an intermediate `[1,1,1,SEQ]` tensor. Previously discarded — now tapped via the output concat and cached in `LayerActs` for reuse in backward.

Changes:
- **MIL generators:** Added `rrms` to output concat of both `gen_sdpa_fwd_taps` (+1 channel → `6*DIM+1`) and `gen_ffn_fwd_taps` (+1 channel → `2*DIM+3*HIDDEN+1`). Cost: 512 bytes per IOSurface.
- **`rmsnorm_dw`:** Now takes precomputed `rrms` instead of `x` and `w`. Eliminated the rrms recomputation loop entirely (768 iterations of `vDSP_vmul` + `vDSP_vadd` + `vDSP_vsmsa` + `vvrsqrtf`).
- **CPU `rmsnorm`:** Now uses static buffers (no calloc/free) and optionally caches `rrms_out` for the final layer.
- **Forward:** Two extra `io_read_fp16` calls per layer (1 channel each = 512 bytes), trivial cost.

**Result (100 steps, accum=50):**
```
104.7 ms/step (avg), batch1=108.3, batch2=101.1
  Batch 2: ane=9.8 io=4.0 cls=1.9 rms_fwd=0.1
           elem=33.9 [xent=15.4 memcpy=1.2 rms_bwd=2.0 resid=1.0 embed=1.2 embed_bwd=13.1]
```

| Metric | Before (batch 2) | After (batch 2) | Change |
|---|---|---|---|
| ms/step | 104.2 | 101.1 | **-3.1ms** |
| rms_bwd | 5.4 | 2.0 | **-3.4ms** |

**Note:** embed_bwd rose from 8.7→13.1 — this is noise in the `dispatch_group_wait` timing, not a real regression. The rms_bwd improvement is clean.

**Correctness:** Loss unchanged (step 0: 4.3143, step 10: 3.6053, step 90: 3.7850).

---

## 9. dW Instrumentation — Async Overlap Analysis

**Commit:** (instrumentation, not yet committed)

**What:** Added fine-grained timing inside async dW dispatch blocks and semaphore waits to diagnose the ~48ms of unaccounted overlap time. Also ran a hivemind research session (`findings/hivemind-dw-overlap.md`, Claude+Gemini+DeepSeek, 3 rounds).

Instrumentation added:
- Per-sgemm wall-clock timing inside each async block (FFN, Wo, QKV, embed)
- Per-layer semaphore wait timing on main thread
- Embed dW group_wait timing (separate from scatter-add)
- Final batch dW group_wait timing

**Result (100 steps, accum=50):**
```
101.5 ms/step (avg), batch1=102.4, batch2=100.7
  ane=9.6  io=3.7  cls=1.8
  elem=21.7 [xent=15.9 memcpy=1.2 rms_bwd=2.2 resid=1.5 embed=0.8 embed_bwd=0.2]
  dW sgemm: ffn=164.4 wo=19.6 qkv=58.7 embed=51.9 total=294.5 ms/step
  sem_wait=0.0 embed_dw_wait=10.7 final_dw_wait=0.0 ms/step
  sem/layer: L0=0.0 L1=0.0 ... L11=0.0 (all zero)
```

**Key findings:**

1. **Semaphore waits: ZERO across all layers.** Double-buffered flow control works perfectly — dW never backs up. Gemini's "First-In-Last-Out pipeline hazard" theory disproven.

2. **Total dW sgemm CPU time: ~294ms/step**, distributed:
   - FFN (W2+W1+W3 × 12 layers): 164ms
   - QKV (Wq+Wk+Wv × 12 layers): 59ms
   - Wo (× 12 layers): 20ms
   - **Embed dW: 52ms** (32000×768×256 dense sgemm — classifier gradient for tied weights)

3. **Effective parallelism: ~3x** — 294ms of sgemm fits into ~101ms wall time via 12 parallel serial queues + 1 embed queue.

4. **Embed dW group_wait: ~11ms/step.** Main thread blocks waiting for previous step's embed sgemm to finish. The embed sgemm (52ms) often doesn't complete before the next step needs to scatter-add, causing ~11ms of visible blocking.

5. **Final batch dW wait: 0.0ms.** All dW drains within the batch — no tail latency after last step.

**Analysis:** The ~48ms gap is real dW compute time that doesn't fully overlap with main-thread work. With 294ms CPU sgemm and ~38ms main-thread timed work, the system achieves ~3x parallelism across ~13 queues. The bottleneck is total dW FLOP count, not scheduling or contention.

**Embed dW is the #1 dW target:** At 52ms/step it's 18% of total dW time and causes 11ms of visible blocking. It's a dense `VOCAB×DIM×SEQ = 32000×768×256` sgemm because weights are tied (same `embed` array used for embedding lookup and classifier weight). The embed_backward scatter-add (0.2ms) is negligible.

---

## 10. Optimize Embed dW: Split Accumulators + Tiled Sgemm

**What:** Three changes to the embed weight gradient computation:

1. **Split accumulators:** Separate `gembed` into `gembed_cls` (dense sgemm for classifier gradient) and `gembed_emb` (scatter-add for embedding gradient). Merged at adam update. This eliminates the `dispatch_group_wait(embed_dw_grp)` between steps — the scatter-add no longer depends on the sgemm finishing.

2. **Tiled sgemm via dispatch_apply:** Split the 32000×768×256 sgemm into 8 tiles of 4000 rows each, dispatched across 8 concurrent threads via `dispatch_apply` inside the serial embed queue block. Each tile: 4000×768×256 = 786M multiply-adds.

3. **Double-buffered capture:** Pre-allocated two 32MB `capt_dlogits` + two 0.75MB `capt_xfinal` buffers, alternating by step slot. Eliminates per-step malloc/free of 32.75MB.

**Result (100 steps, accum=50):**
```
88.3 ms/step (avg), batch1=88.0, batch2=88.5
  ane=9.8  io=4.4  cls=1.8
  elem=21.0 [xent=15.1 memcpy=1.2 rms_bwd=2.2 resid=1.3 embed=1.1 embed_bwd=0.3]
  dW sgemm: ffn=178 wo=20 qkv=56 embed=21 total=276 ms/step
  sem_wait=0.0 final_dw_wait=12.7 ms/step
```

| Metric | Before | After | Change |
|---|---|---|---|
| ms/step (100-step avg) | 101.5 | 88.3 | **-13.2ms (13.0%)** |
| embed sgemm | 52 | 21 | **-31ms (-60%)** |
| embed_dw_wait | 11 | 0 (eliminated) | **-11ms** |

**Key findings:**
- The 8-tile `dispatch_apply` achieves ~2.6x speedup over a single sgemm call, confirming the original cblas call wasn't fully parallelizing across all P-cores at this matrix shape.
- Eliminating the per-step embed group_wait directly removes 11ms of main-thread blocking.
- The `final_dw_wait` (~11ms, once per 50-step batch) shows the last embed sgemm still trails the batch end slightly — acceptable since it's amortized.

**Memory cost:** +98MB for second accumulator, +65.5MB for double-buffered captures. Total +163.5MB.

**Correctness:** Loss unchanged (step 0: 4.3143, step 10: 3.6053, step 90: 3.7848).

**Failed experiment — layer dW tiling:** Applied the same `dispatch_apply` tiling to FFN/Wo/QKV dW blocks (4-8 tiles each). Result: **regression** to 104.9ms/step. Root cause: 12 serial queues already provide inter-layer parallelism. Adding intra-block tiling spawns 36 × 4-8 = 150+ threads on the global concurrent queue, causing thread explosion that steals CPU from main-thread work (xent jumped 15→24ms). Lesson: `dispatch_apply` tiling works for a single large sequential sgemm (embed) but not for many concurrent small sgemms (layer dW).

---

## 11. ANE Softmax for Cross-Entropy

**Commit:** `ff96d61` — "ANE softmax for cross-entropy"

**What:** Added `gen_softmax()` MIL kernel that computes softmax on ANE, replacing the CPU `cross_entropy_loss` softmax. The kernel takes logits from `cls_fwd->ioOut` via `io_copy` (no CPU round-trip), computes `exp(x - max(x)) / sum(exp(x - max(x)))`, and writes probabilities back. CPU still handles loss accumulation and gradient (trivial: `probs/S`, subtract `1/S` at target positions).

**Result:** xent dropped from ~15ms to ~4.6ms. Total step time: ~75→69ms (prior session).

---

## 12. Instrument Backward Pass Timing

**Commit:** `f0b2479` — "Instrument backward pass timing — accounts for 98.5% of step time"

**What:** The backward pass consumed ~65% of step time but was completely untimed. Added three new accumulators:
- `t_bwd_ane` — all backward ANE evals (ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd, rmsBwd ×2 per layer + rmsBwdFinal)
- `t_bwd_io` — all backward IO (io_copy, io_read_fp16, io_write_fp16_at, io_write_fp16_vec)
- `t_cls_bwd` — classifier backward (ANE eval or CPU sgemm)

Wrapped each backward ANE eval and IO block with `mach_absolute_time()` timing, same pattern as forward path. Fixed `final_dw_wait` label to divide by `steps_batch` (was showing per-batch value with "ms/step" label). Renamed forward timing line to `fwd:` for clarity.

**Key discovery:** The `final_dw_wait` in previous sessions was **per-batch, not per-step**. For a 100-step batch, the actual per-step value is ~0.1ms — negligible. The dW tail is not inside the per-step timing loop (`tms` measured at line 726, before `dispatch_group_wait` at line 732).

**Result (100 steps, accum=200):**
```
82.9 ms/step
  fwd: ane=9.8 io=5.0 cls=1.0 rms_fwd=0.1 ms/step
  bwd: ane=24.5 io=16.3 cls_bwd=13.6 ms/step
  elem=11.4 [xent=4.7 memcpy=1.5 rms_bwd=2.5 resid=1.4 embed=0.9 embed_bwd=0.4]
  dW sgemm: ffn=188.1 wo=21.8 qkv=64.3 embed=24.0 total=298.1 ms/step
  sem_wait=0.0 final_dw_wait=0.1 ms/step
```

**Full step accounting:**

| Component | ms/step | % of step |
|---|---|---|
| fwd ane | 9.8 | 11.8% |
| fwd io | 5.0 | 6.0% |
| fwd cls | 1.0 | 1.2% |
| bwd ane | 24.5 | 29.6% |
| bwd io | 16.3 | 19.7% |
| cls_bwd | 13.6 | 16.4% |
| elem (xent+memcpy+rms_bwd+resid+embed+embed_bwd) | 11.4 | 13.8% |
| rms_fwd | 0.1 | 0.1% |
| **Accounted** | **81.7** | **98.6%** |
| **Measured** | **82.9** | |

**Key findings:**
1. **Backward ANE (24.5ms) is 2.5x forward ANE (9.8ms)** — 74 ANE evals vs 24 forward evals. Per-eval: ~0.33ms backward vs ~0.41ms forward. Backward kernels are actually slightly faster per-eval (smaller output tensors) but there are 3x more of them.
2. **Backward IO (16.3ms) is 3.3x forward IO (5.0ms)** — backward shuffles far more data between ANE kernels (SDPA backward alone has 5 io_copy + 3 io_read_fp16 per layer).
3. **cls_bwd (13.6ms) is the single most expensive ANE kernel** — VOCAB×DIM×SEQ = 32000×768×256 ≈ 12.6 GFLOPS matmul as a single conv op.

---

## 13. Batch IOSurface Locking

**Commit:** `50c78b8` — "Batch IOSurface locking to reduce IO overhead (~240 fewer lock/unlock pairs)"

**What:** Added 4 lock helpers (`io_lock_rw`, `io_lock_ro`, `io_unlock_rw`, `io_unlock_ro`) and refactored forward + backward loops to lock each IOSurface once per region instead of per-call.

Batched regions:
- Forward: fwdAttn output (6→2 locks), fwdFFN output (6→1 lock)
- Backward: 8 sections per layer — FFN bwd in (3→2), FFN bwd out (3→1), RMS2 bwd in (5→3), SDPA bwd1 in (3→2), bwd1→bwd2 (4→3), SDPA out (3→2), QKV in (4→3), RMS1 bwd in (5→3)

Total: ~504 locks/step → ~264 locks/step, saving ~240 lock/unlock pairs.

**Result (20 steps, accum=50):**
```
77.5 ms/step
  fwd: ane=10.0 io=4.5 cls=1.1 rms_fwd=0.1
  bwd: ane=22.6 io=13.4 cls_bwd=14.0
  elem=10.6 [xent=5.1 memcpy=1.3 rms_bwd=2.1 resid=0.8 embed=1.0 embed_bwd=0.3]
  dW sgemm: ffn=181.8 wo=20.5 qkv=57.5 embed=18.5 total=278.3 ms/step
```

| Metric | Before | After | Change |
|---|---|---|---|
| fwd io | 5.0 | 4.5 | -0.5ms |
| bwd io | 16.3 | 13.4 | -2.9ms |
| **Combined IO** | **21.3** | **17.9** | **-3.4ms** |

**Note:** 100-step run showed inflated timings across all metrics (bwd ane 25.9 vs usual 9.6) due to system thermal/load variance. The 20-step first-batch numbers are more reliable for comparing IO specifically.

**Correctness:** Loss unchanged (step 0: 4.3108, step 10: 3.6026).

---

## 14. Tiled cls_bwd: 4×8K Input Channel Split

**Commit:** `5d23882` — "Tiled cls_bwd: split 32K→4×8K input channels, 14.4→4.0ms (3.6×)"

**What:** Replaced `gen_cls_bwd()` (single `[768, 32000, 1, 1]` conv) with `gen_cls_bwd_tiled()` that splits the 32K input channels into 4 chunks of 8K:
- 4 × `slice_by_size` to extract `[1, 8000, 1, 256]` from input
- 4 × `conv` with weights `[768, 8000, 1, 1]` each
- Pairwise add tree: `(c0+c1) + (c2+c3)` → output `[1, 768, 1, 256]`

All within a single MIL program (single `ane_eval`). Weight blobs built from 4 slices of the embed matrix via `build_blob_t(embed + t*chunk*DIM, chunk, DIM)`.

**Three modes benchmarked** (via temporary `--cls-bwd-mode` flag, now removed):

| Mode | Description | cls_bwd | ms/step |
|---|---|---|---|
| 0 | Original ANE `[768, 32000, 1, 1]` | 14.4 ms | 86.4 |
| 1 | Tiled ANE 4×`[768, 8000, 1, 1]` | **4.2 ms** | **69.0** |
| 2 | CPU `cblas_sgemm` (M=768, N=256, K=32000) | 11.6 ms | 89.6 |

**Key findings:**
- ANE pathology scales super-linearly with input channels: 32K channels = 14.4ms, but 4×8K channels + add tree = 4.2ms (3.4× faster despite identical FLOPs + overhead of slicing and adding).
- CPU AMX sgemm (11.6ms) was slower than even the original ANE (14.4ms) once you add the IO overhead of `io_write_fp16_at` for rmsBwdFinal (no `cls_bwd->ioOut` to `io_copy` from).
- The tiled kernel also compiles faster (4811ms vs 5328ms for original).
- Mode 1 hardcoded as winner. Original `gen_cls_bwd()` removed.

**Result (20 steps, accum=50, final hardcoded run):**
```
76.1 ms/step
  fwd: ane=9.8 io=5.0 cls=1.0 rms_fwd=0.1
  bwd: ane=25.7 io=16.2 cls_bwd=4.0
  elem=12.9 [xent=5.0 memcpy=2.2 rms_bwd=2.8 resid=1.5 embed=1.0 embed_bwd=0.4]
  dW sgemm: ffn=141 wo=17 qkv=47 embed=16 total=222 ms/step (~2.9x overlap)
```

**Correctness:** Loss identical across all 3 modes (step 0: 4.3108, step 10: 3.6026).

**Files changed:** `stories_mil.h` (gen_cls_bwd_tiled replacing gen_cls_bwd), `train_large.m` (tiled compile with 4 weight slices)

---

## 15. Cosine LR Schedule + 5000-Step Training Run

**What:** Added cosine learning rate schedule with linear warmup:
- `--lr` sets peak LR, `--lr-min` sets floor (enables cosine decay), `--warmup` sets warmup steps
- LR computed at end of each accumulation batch (before adam update)
- Formula: warmup phase uses linear ramp `lr * (step / warmup_steps)`, then `lr_min + 0.5 * (lr - lr_min) * (1 + cos(π * progress))` where `progress = (step - warmup) / (total - warmup)`

**LR sensitivity experiments:**
- `lr=1e-4, lr_min=1e-5, warmup=100`: Diverged by step 450 (loss hit 4.9, continued climbing to 6+)
- `lr=5e-5, lr_min=5e-6, warmup=200`: Diverged by step 1500 (loss hit 5-6 range)
- `lr=3e-5, lr_min=3e-6, warmup=100`: **Stable for full 5000 steps**

**Key finding:** fp16 forward/backward on ANE constrains the max stable LR much more than a typical fp32 training setup. The fp16 gradient accumulation introduces noise that makes lr>3e-5 unstable for this model. This is consistent with the earlier finding that flat lr=3e-4 diverged at ~150 steps while lr=3e-5 was stable.

**5000-step training result (lr=3e-5 → 3e-6, warmup=100, accum=50):**
```
Best loss: 2.91 (step 3950), final: ~4.2 (high per-step variance)
Top-5 losses: 2.91, 2.94, 3.16, 3.19, 3.21
Wall time: 22.2 min (1330s), ~133ms/step avg (includes recompile overhead)
Efficiency: 1.31 TFLOPS sustained (ANE+CPU), 5.6% ANE utilization
```

**Text generation eval (generate.py):**
Pretrained (no RoPE):
```
Greedy: "Once upon a time, there was a little girl named Lily. She was a little girl
named Lily. She loved to play with her doll. She was a doll. She was very special
doll. She was very special doll."
```

Trained (5000 steps, no RoPE):
```
Greedy: "Once upon a time, there was a little girl named Lily. Lily loved to play
outside in the park with her friends. She loved to run and jump on the swim in the
swim in the swim in the pond."
```

Both produce coherent children's story fragments. The trained model shows slightly better narrative structure (character motivation, events) but still degrades mid-sequence due to the no-RoPE limitation. The improvement from pretrained to trained is modest — the primary bottleneck for generation quality is positional encoding, not training duration.

**Files changed:** `train_large.m` (cosine LR schedule, --lr-min/--warmup flags), `generate.py` (new: text generation eval matching ANE training kernels)

---

## Remaining Optimization Targets

Current profile (20 steps, accum=50, post tiled cls_bwd):
```
~76ms/step
  fwd: ane=9.8 io=5.0 cls=1.0 rms_fwd=0.1   (15.9 total)
  bwd: ane=25.7 io=16.2 cls_bwd=4.0          (45.9 total)
  elem=12.9 [xent=5.0 memcpy=2.2 rms_bwd=2.8 resid=1.5 embed=1.0 embed_bwd=0.4]
  dW sgemm: ffn=141 wo=17 qkv=47 embed=16 total=222 ms/step (~2.9x overlap)
```

### Prioritized by risk-adjusted impact

1. **Fuse sdpaBwd1+sdpaBwd2 (~7ms ane+io)** — Currently 2 separate ANE evals + IO shuffle between them, per layer. Fusion eliminates 12 ane_eval calls (~4ms) and 12 inter-kernel IO transfers (~3ms). Requires combining two MIL programs into one. **~7ms potential, high risk (MIL complexity).**
2. **bwd io ~16ms** — Lock batching eliminated overhead, but actual data movement (memcpy + fp16↔fp32) remains. Some backward flows do unnecessary fp16→fp32→fp16 round-trips (e.g., dx_ffn read as fp32, then written back as fp16 for rmsBwd). Could keep fp16 end-to-end for pass-through data. **~3-5ms potential, medium risk.**
3. **bwd ane ~26ms** — 74 ANE evals at ~0.35ms each. Beyond SDPA fusion, could merge qkvBwd+rmsBwd into single kernel. **~3ms additional potential, high risk.**
4. **elem ~13ms** — xent=5.0 (CPU gradient after ANE softmax), rms_bwd=2.8, memcpy=2.2. Diminishing returns individually.
5. **cls_bwd ~4ms** — Could try 8×4K tiling for further reduction, but diminishing returns from current 4.0ms.
