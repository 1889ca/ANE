# ANE Training: Possible Futures

Where this work could go, and what "useful" means for on-device ANE training.

## What We Have (March 2026)

- First known transformer training loop running on Apple Neural Engine
- 110M parameter model (Stories110M) training at ~74ms/step on M4 consumer hardware
- Full pipeline: ANE forward/backward via MIL→CoreML, CPU weight updates (Adam), checkpointing, text generation eval
- 2.6x optimization from naive baseline (190→74ms/step) across 19 incremental changes
- Novel reverse engineering: IOSurface memory model, zero-copy ANE↔GPU aliasing, MIL training kernels
- Branch: `cls-ane-kernel`, all code in `training/`

## Here-and-Now Useful Things

### 1. On-device fine-tuning with privacy guarantee

The most immediate practical use. Take a pretrained model, fine-tune on private data (notes, code style, domain docs) without data ever leaving the device. The selling point is privacy, not speed — no M4 beats an H100, but no H100 can promise your data stays local.

At 74ms/step with seq=256, a 1000-step fine-tune takes ~75 seconds. That's already usable if the base model is useful enough for narrow tasks.

### 2. Reference implementation / proof-of-concept

This is genuinely novel work. The ANE training pathway through MIL, the IOSurface memory model, the compilation-as-weight-update pattern, the zero-copy discoveries — none of this is documented anywhere else. A paper or detailed writeup has standalone value for the Apple ML research community.

### 3. Distillation target

Use a large model (via API) to generate labeled training data, then distill into a small on-device model. The training loop makes the last mile local: query Claude/GPT for examples, train a tiny task-specific model that runs entirely offline.

## If We Could Do X, Then Y

### Batching (batch=4-8)

**Blocker:** IOSurface layout fixes sequence dimension at compile time. Batch dimension must be baked in, multiplying surface sizes.

**Engineering, not fundamental.** The MIL programs already operate on [C, 1, S, 1] tensors. Extending to [C, 1, S, B] or [C, B, S, 1] requires adjusting surface allocation and kernel generation, but the math is identical.

**Unlock:** 3-4x throughput. Fine-tuning 110M in ~20 seconds. Makes interactive fine-tune-and-test loops practical.

### Avoid recompilation

**Blocker:** ANE has no writable weight registers via public APIs. Every weight update requires recompiling MIL→ANE executables. Currently 45% of wall time.

**Paths:** (a) Reverse-engineer direct weight DMA writes (high risk, Apple could break it), (b) Apple exposes training-friendly API (unlikely near-term), (c) accumulate more steps between recompiles (diminishing returns on convergence).

**Unlock:** ~2x throughput immediately. Training speed becomes compute-bound rather than compile-bound.

### Scale to 350M-1B parameters

**Blocker:** Larger models need more IOSurface memory and longer compile times. The ANE has the TFLOPS (15.8 theoretical, we use ~10%) — bottleneck is IO and compilation, not compute.

**Key insight:** Larger matmuls have better compute/IO ratio. A 350M model would amortize per-layer overhead over more FLOPs per operation. ANE utilization should increase significantly.

**Unlock:** Models that are actually useful for narrow tasks. 110M is a proof of concept; 350M-1B is where language models start being genuinely helpful for classification, summarization, code completion on constrained domains.

### Batching + larger model + compilation caching (the big unlock)

Train on ANE while using your computer normally. Apple Silicon shares GPU between display and compute — running training on ANE leaves GPU free for rendering/UI. `mlx train` eats your GPU; `ane-finetune` doesn't.

**Unlock:** Background fine-tuning that doesn't tank your UI. "Train while you work" on a MacBook.

## Most Realistic Product: `ane-finetune`

A CLI tool that takes a base model + text corpus and produces a fine-tuned checkpoint, entirely on-device.

**Target audience:** Privacy-conscious developers who want small task-specific models.

**Moat:** "Your data never leaves your machine" + "doesn't eat your GPU."

**Biggest gaps to close:**
1. Batching — for practical training speed
2. Model quality — 110M may be too small for real tasks; need to validate or scale up
3. Tokenizer + data pipeline — current implementation is tightly coupled to TinyStories
4. Export — fine-tuned model needs to be usable (CoreML export? GGUF? Direct inference?)

See `ane-finetune.md` for product design exploration.
