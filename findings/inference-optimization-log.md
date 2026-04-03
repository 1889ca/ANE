# Bonsai-8B Inference on Apple Silicon: ANE + Metal Hybrid

## Overview

Running a 1-bit 8B parameter LLM (Bonsai-8B, Qwen3 architecture) on Apple Silicon using a hybrid Apple Neural Engine + Metal GPU pipeline. Starting from a pure ANE implementation at 0.3 tok/s, optimized to 2.5 tok/s through 11 experiments across two days.

**Model:** Bonsai-8B (PrismML) — Q1_0_g128 quantized, 36 layers, 4096 dim, 12288 FFN, 32 heads, 8 KV heads (GQA), 151669 vocab

**Hardware:** Apple M4

**Final architecture:** ANE runs QKV projections, Metal GPU runs Wo + RMSNorm + FFN (fused), CPU handles attention decode + classifier + RMSNorm + RoPE

## Key Findings

### 1. ANE baked-weight kernels have high per-eval overhead

ANE compiles MIL programs with weights embedded ("baked") into the kernel binary. Each `ane_eval()` reloads all weights from memory. For Stories110M (768-dim, ~3MB per kernel), this is fast (~0.5ms). For Bonsai-8B FFN (~300MB per kernel), it takes **199ms per eval** — completely dominated by weight loading, not compute.

**Lesson:** ANE baked weights are designed for batch inference (many tokens, amortized weight load). Single-token autoregressive decode is the worst case.

### 2. Metal GPU is faster than ANE for large weight matrices

Metal compute shaders with shared memory buffers stream weights from unified memory efficiently. A simple `simd_sum` matvec kernel (32 threads per row, half4 vectorized loads) handles 300MB FFN weights in ~8ms vs ANE's 199ms.

**Lesson:** For weight-bound operations (matvec with large W), Metal beats ANE by ~25x. ANE's advantage is in compute-bound operations with small weights.

### 3. ANE beats Metal for small projections

QKV projection (80MB weights: Wq + Wk + Wv) and Wo (32MB) run faster on ANE (~35ms, ~23ms) than equivalent Metal command buffers. Metal's `commit + waitUntilCompleted` overhead (~5-10ms per submission) outweighs the compute advantage for these smaller matrices.

**Lesson:** Choose ANE vs Metal per-operation based on weight size. Crossover is roughly at ~100MB.

### 4. Fusing Metal dispatches into one command buffer is critical

Each Metal command buffer submission has ~5-10ms overhead (encode, commit, wait). By fusing Wo projection + residual add + RMSNorm + FFN (7 GPU dispatches) into a single command buffer, we eliminate 36 extra sync points per token.

**Lesson:** Minimize `commit + waitUntilCompleted` calls. Encode as many dispatches as possible into each command buffer, even if they're different operations.

### 5. fp16 overflow requires fp32 residual stream for 1-bit models

1-bit (Q1_0_g128) weights produce hidden states that grow beyond fp16 range (~65504) by layer 27 of 36. The residual stream accumulates +-scale contributions that don't cancel as well as fp16/fp32 weights.

**Solution:** Maintain residual stream in fp32 on CPU. Convert to fp16 only for ANE/Metal kernel input (after RMSNorm, which normalizes to small values). RMSNorm must happen in fp32 to see the true magnitude before normalizing.

**Lesson:** Don't fuse residual adds into ANE/Metal kernels for 1-bit models. Do them on CPU in fp32.

### 6. ANE needs minimum spatial dimension of 32

ANE kernels with spatial dimension < 32 (e.g., S=1, S=4, S=16) fail at runtime with `ANEProgramProcessRequestDirect() Failed`. Compilation succeeds but eval returns zeros. S=32 is the minimum that works reliably.

**Workaround:** Use DECODE_S=32 and zero-pad. For single-token decode, data goes at position 0 of a [C, 32] tensor with positions 1-31 unused. This wastes ~32x spatial compute but ANE throughput is high enough that it doesn't matter.

### 7. AMX simdgroup_matrix doesn't help for matvec

Apple's AMX instructions (`simdgroup_matrix_multiply_accumulate`) operate on 8x8 tiles. For matrix-vector multiply (M x 1), the x vector must be broadcast to 8x8 via threadgroup shared memory. The broadcast overhead (threadgroup writes + barriers, 512 times per row for K=4096) makes AMX **7x slower** than simple `simd_sum` for matvec.

**Lesson:** AMX is for M×N×K matmul where all dimensions are large. For matvec (N=1), scalar SIMD reduction is optimal.

### 8. Metal classifier is slower than CPU NEON for large vocab

151K vocabulary × 4096 dim classifier requires 151K threadgroups on Metal. The GPU command buffer overhead for this many threadgroups exceeds CPU `dispatch_apply` with NEON fp16 dot products (761ms Metal vs 484ms CPU).

**Lesson:** CPU parallel NEON with `dispatch_apply` (chunked to 256 rows per dispatch unit) beats Metal for vocab-scale operations where each row's compute is small.

### 9. Async ANE+Metal overlap doesn't help on unified memory

Attempted launching Metal Wo+FFN async while ANE runs QKV for the next layer. Result: 44% slower. Likely cause: unified memory bandwidth contention — ANE and Metal GPU share the same memory bus, so concurrent weight loading doesn't pipeline.

**Lesson:** On Apple Silicon unified memory, ANE and Metal GPU are bandwidth-competitors, not independent pipelines. Serialize for best throughput.

### 10. Qwen3 QK norms are essential

Qwen3 applies per-head RMSNorm (dim=128) to Q and K vectors after projection, before RoPE. Without QK norms, attention scores explode and the model produces garbage. These are stored as `attn_q_norm.weight` and `attn_k_norm.weight` (F32) in GGUF.

### 11. GGUF tokenizer parsing is straightforward

The GGUF file embeds the full BPE tokenizer in metadata KV pairs:
- `tokenizer.ggml.tokens`: array of 151K strings
- `tokenizer.ggml.merges`: array of ~151K merge rules ("token_a token_b")
- BPE encode: start with bytes, iteratively merge lowest-rank pairs

A hash table for merge rank lookup (262K buckets) makes BPE encoding fast enough to be negligible.

## Optimization Timeline

| # | Change | tok/s | ms/tok | Speedup | Notes |
|---|--------|-------|--------|---------|-------|
| 0 | Baseline (all ANE) | 0.3 | 3478 | 1.0x | 108 ANE evals/tok |
| 1 | Parallel NEON classifier | 0.3 | 3234 | 1.1x | dispatch_apply, chunked 256 |
| 2 | ANE Wo kernel | 0.8 | 1308 | 2.7x | Moved Wo from CPU to ANE |
| 3 | Metal FFN | 1.7 | 583 | 6.0x | 199ms ANE → ~8ms Metal |
| 4 | Simdgroup matvec | 2.1 | 476 | 7.3x | 32-thread simd_sum, half4 |
| 5 | FFN-only Metal buffers | 2.3 | 433 | 8.0x | Reduced memory pressure |
| 6 | Metal classifier | -- | -- | -- | **Reverted** (CPU faster) |
| 7 | Multi-row threadgroups | -- | -- | -- | **Reverted** (broken) |
| 8 | AMX simdgroup_matrix | -- | -- | -- | **Reverted** (7x slower) |
| 9 | All-Metal QKV | -- | -- | -- | **Reverted** (cmd buf overhead) |
| 10 | Fused Wo+RMS+FFN | 2.5 | 399 | 8.7x | 7 dispatches, 1 sync |
| 11 | Async ANE+Metal | -- | -- | -- | **Reverted** (bandwidth contention) |

## Final Performance Breakdown

Per token (399ms average):
- **ANE QKV:** ~120ms (36 evals × ~3.3ms) — 30%
- **Metal Wo+RMS+FFN:** ~240ms (36 cmd bufs × ~6.7ms) — 60%
- **CPU classifier:** ~15ms — 4%
- **CPU attention:** ~8ms — 2%
- **CPU other (RMSNorm, RoPE, I/O):** ~16ms — 4%

## Theoretical Limits

- Weights per token: ~412MB/layer × 36 layers = 14.8 GB
- M4 memory bandwidth: ~100 GB/s
- Bandwidth-limited minimum: ~148ms/tok
- Current: 399ms/tok = **2.7x theoretical**
- Main gap: dispatch/sync overhead (~36 ANE evals + 36 Metal cmd bufs)

## Architecture Diagram

```
Per token, per layer:

  [fp32 residual x]
        |
  CPU RMSNorm(att) → fp16
        |
  ANE QKV kernel ──→ Q[4096], K[1024], V[1024]  (fp16)
        |
  CPU QK norm + RoPE + KV cache append
        |
  CPU attention decode (scalar, fp16 KV cache)
        |                          ┌─────────────────────┐
  Metal command buffer (1 sync):   │ Wo matvec            │
        |                          │ residual_add (x+Wo)  │
        |                          │ RMSNorm(ffn)         │
        |                          │ W1 matvec (gate)     │
        |                          │ W3 matvec (up)       │
        |                          │ silu_gate            │
        |                          │ W2 matvec (down)     │
        |                          └─────────────────────┘
        |
  CPU fp32 residual add: x += wo_delta + ffn_delta
        |
  [fp32 residual x] → next layer
```

## File Map

- `infer.m` — Main inference binary, CLI, decode loop
- `infer_config.h` — Model config structs (Bonsai-8B, Stories110M)
- `infer_mil.h` — ANE MIL kernel generators (QKV, Wo, FFN, RMSNorm)
- `infer_metal.h` — Metal GPU pipeline (matvec, silu_gate, residual_add, rmsnorm)
- `infer_tokenizer.h` — GGUF BPE tokenizer parser
- `infer_xor_patch.h` — Bankai XOR patch support for 1-bit weight modification
- `kv_cache.h` — fp16 KV cache (channel-first layout)
- `bonsai_lora.h` — GGUF loader, Q1 dequant, LoRA merge

## Build & Run

```bash
cd training && make infer

# Stories110M (ANE-only, 150 tok/s)
./infer --stories --model stories110M.bin --prompt "Once upon a time" --max-tokens 128

# Bonsai-8B (ANE+Metal hybrid, 2.5 tok/s)
./infer --model Bonsai-8B.gguf --prompt "The meaning of life is" --max-tokens 64 --temp 0.7

# With LoRA adapter
./infer --model Bonsai-8B.gguf --lora style-adapter.gguf --prompt "text"

# With Bankai XOR patch
./infer --model Bonsai-8B.gguf --xor patch.json --prompt "text"
```
