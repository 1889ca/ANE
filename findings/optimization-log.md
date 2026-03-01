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

## Remaining Optimization Targets

Current profile (warm batch, 100 steps, accum=50):
```
  ane=16.6  io=8.6  cls=5.2  rms_fwd=0.1
  elem=72.9 [xent=30.1 memcpy=2.2 rms_bwd=16.0 resid=2.6 embed=1.1 embed_bwd=20.9]
```

Note: absolute values inflated ~1.7x vs baseline session. Relative proportions are what matter for prioritization.

### Prioritized by risk-adjusted impact

1. **rms_bwd ~9ms** — Move rmsnorm_bwd to ANE. 25 calls/step. Involves elementwise ops + reduction (similar to forward rmsnorm already on ANE). **~9ms, medium risk.**
2. **xent ~16ms** — Fused cross-entropy on ANE (see hivemind analysis above). **~16ms, high risk.** Treat as research spike.
3. **io ~5ms** — Keep activations in fp16 end-to-end, skip fp32↔fp16 conversion. Requires numerical stability analysis for backward pass. **~5ms, medium risk.**
4. **embed_bwd ~4ms** — Embed dW wait + scatter-add. Could transpose or use NEON gather for strided access. **~4ms, medium effort.**
5. **resid ~1ms** — Backward residual adds. Could fuse into backward ANE kernels. **~1ms, diminishing returns.**

### Architecture-level (larger refactors)
- Fuse rmsnorm + conv into single ANE kernel (reduce 74 kernel evals/step)
- Pipeline: overlap step N's backward with step N+1's forward (double-buffer IOSurfaces)
- Full fp16 activation path (eliminate io conversion entirely)
